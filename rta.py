import numpy as np
from sklearn.externals import joblib
import pandas as pd
from functools import partial
import os
import sqlite3
import time
import yaml
from operator import le, lt, eq, ne, ge, gt

from astropy.coordinates import AltAz, SkyCoord
import astropy.units as u
from astropy.table import Table

from fact.instrument.constants import LOCATION
from fact.analysis.source import calc_theta_camera, calc_theta_offs_camera
from fact.coordinates import camera_to_equatorial, horizontal_to_camera

from klaas.apply import predict_energy, predict_disp, predict_separator
from klaas.parallel import parallelize_array_computation
from klaas.configuration import KlaasConfig
from klaas.feature_generation import feature_generation
from klaas.preprocessing import calc_true_disp

#Paths
gammalimit = 0.84 #Gamma-Cut = 0.84
thetalimit = 1.0 #Theta-Cut for db = 0.
thetalimitexcess = 0.16 #Theta-Cut for excess rate = 0.16
timeouttime = 5 
configuration_path = 'aict.yaml'
quality_cuts_path = 'quality_cuts.yaml' 
separator_model_path = 'separator.pkl' 
energy_model_path = 'energy.pkl'
disp_model_path = 'disp.pkl'
sign_model_path = 'sign.pkl'
data_path = 'Data/'
data_remove_path = 'Data/done/'
aux_path = 'Data/AUX/' 
database_path = 'Online/rta.db'
fact_sources = 'fact_sources.csv'
    
#Needed features
needed_columns = [
        'cog_x', 
        'cog_y', 
        'delta', 
        'pointing_position_az', 
        'pointing_position_zd', 
        'run_id', 'event_num',
        ]

#features for Database
db_features = [
    'night',
    'run_id',
    'event_num',
    'timestamp',
    'gamma_energy_prediction',
    'gamma_prediction',
    'ra_prediction',
    'dec_prediction',
    'theta_deg',
    'theta_deg_off_1',
    'theta_deg_off_2',
    'theta_deg_off_3',
    'theta_deg_off_4',
    'theta_deg_off_5',
    'source',
]

OPERATORS = {
    '<': lt, 'lt': lt,
    '<=': le, 'le': le,
    '==': eq, 'eq': eq,
    '=': eq,
    '!=': ne, 'ne': ne,
    '>': gt, 'gt': gt,
    '>=': ge, 'ge': ge,
}

#Load .fit File and start analysis 
def loadfits():
    filename = [f for f in os.listdir('Data/') if f.endswith('.fits') or f.endswith('.fit')]
    for i in range(len(filename)):
        analysis(data_path + filename[i])
        print('Processed ' + filename[i] +' successfully.')
        os.rename(data_path + filename[i], data_remove_path + filename[i])
    time.sleep(timeouttime) 
    loadfits()

#Apply ML Models
def analysis(file_path):
    #Load paths
    config = KlaasConfig.from_yaml(configuration_path)
    separator_model = joblib.load(separator_model_path)
    energy_model = joblib.load(energy_model_path)
    disp_model = joblib.load(disp_model_path)
    sign_model = joblib.load(sign_model_path)
    t = Table.read(file_path) 
    dd = t.to_pandas() #Create PandasDataframe
    
    #Apply Qualitycuts
    with open(quality_cuts_path) as f:
        quality_cuts = yaml.safe_load(f)

    quality_cuts_selection = quality_cuts.get('selection', {})    
    quality_cuts_mask = np.ones(len(dd), dtype=bool)

    for name, (operator, value) in quality_cuts_selection.items():
        ncut = OPERATORS[operator](dd[name], value)
        quality_cuts_mask = np.logical_and(quality_cuts_mask, ncut)
    
    df = dd.loc[quality_cuts_mask]
    df.index = pd.RangeIndex(len(df.index))

    columns = set(needed_columns)
    
    for model in ('separator', 'energy', 'disp'):
        model_config = getattr(config, model)
        columns.update(model_config.columns_to_read_apply)
    try:
        sources = df['source'].unique()
        source = SkyCoord.from_name(sources[0])
        columns.update(['timestamp', 'night'])
    except (KeyError, OSError) as e:
        source = None
    
    
    #Gamaness prediction
    df_sep = feature_generation(df, config.separator.feature_generation)
    df['gamma_prediction'] = predict_separator(
            df_sep[config.separator.features], separator_model,
    )

    
    #DISP prediction
    df_disp = feature_generation(df, config.disp.feature_generation)
    disp = predict_disp(
        df_disp[config.disp.features], disp_model, sign_model
    )
    
    
    #Energy prediction
    df_energy = feature_generation(df, config.energy.feature_generation)
    df['gamma_energy_prediction'] = predict_energy(
    df_energy[config.energy.features], energy_model,
    )

    #calculate disp_prediction
    source_x = df.cog_x + disp * np.cos(df.delta)
    source_y = df.cog_y + disp * np.sin(df.delta)
    df['source_x_prediction'] = source_x
    df['source_y_prediction'] = source_y
    df['disp_prediction'] = disp
    
    #decode timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'].str.decode('ascii'))
    
    #Parallelized Computation
    if source:
        source_altaz = concat_results_altaz(parallelize_array_computation(
            partial(to_altaz, source=source),
            df['timestamp'],
        ))
        result = parallelize_array_computation(
            calc_source_features_obs,
            source_x,
            source_y,
            source_altaz.zen.deg,
            source_altaz.az.deg,
            df['pointing_position_zd'].values,
            df['pointing_position_az'].values,
            df['timestamp'],
        )
    else:
        result = parallelize_array_computation(
            calc_source_features_sim,
            source_x,
            source_y,
            df['source_position_zd'].values,
            df['source_position_az'].values,
            df['pointing_position_zd'].values,
            df['pointing_position_az'].values,
            df['cog_x'].values,
            df['cog_y'].values,
            df['delta'].values,
        )
    #Merge results together    
    for k in result[0].keys():
            df[k] = np.concatenate([r[k] for r in result])
    
    #RA and Dec prediction
    df['ra_prediction'], df['dec_prediction'] = camera_to_equatorial(
        df['source_x_prediction'],
        df['source_y_prediction'],
        df['pointing_position_zd'],
        df['pointing_position_az'],
        df['timestamp'],
    )
    
    #Source from Exel Table
    df['source'] = sourcename(df.source_position_x[0], df.source_position_y[0], df.pointing_position_zd[0], df.pointing_position_az[0], df.timestamp[0])
    
    timefrom = df.timestamp[0]
    timeto = df.timestamp[len(df.timestamp)-1] 
    timebin = (timeto - timefrom).total_seconds() / 3600.0
    
    
    #Auswahl der Events
    mask_gamma = df['gamma_prediction'] >= gammalimit
    mask_theta = df['theta_deg'] <= thetalimit
    for i in range(1, 6):
        mask_theta |= df[f'theta_deg_off_{i}'] <= thetalimit
    selected = df.loc[mask_gamma & mask_theta, db_features]
    
    #CalculateExess
    num_theta = selected['theta_deg'] <= thetalimitexcess
    for i in range(1, 6):
        num_theta_off = selected[f'theta_deg_off_{i}'] <= thetalimitexcess
    on = len(selected.loc[num_theta, db_features])
    off = len(selected.loc[num_theta_off, db_features])
    rate = (on - off*0.2)/(timebin)
    excess = pd.DataFrame({'source': df.source[0], 'night' : df.night[0], 'run_id' : df.run_id[0], 'rate' : [rate], 'timeto' : [timeto], 'timefrom':[timefrom], 'n_on' : on, 'n_off' : off})
    sql(selected, excess)

def sql(selected, excess):
    conn = sqlite3.connect(database_path)  
    cur = conn.cursor()  
    cur.execute('''CREATE TABLE IF NOT EXISTS events (dec_prediction real, event_num integer, gamma_energy_prediction real, gamma_prediction real, night integer, ra_prediction real, run_id integer, theta_deg real, theta_deg_off_1 real, theta_deg_off_2 real, theta_deg_off_3 real, theta_deg_off_4 real, theta_deg_off_5 real, timestamp text, source text)''')    
    selected.to_sql(name="events", con=conn, if_exists="append", index=False)
    cur.execute('''CREATE TABLE IF NOT EXISTS run (source text, night integer, run_id integer, rate real, timefrom text, timeto text, n_on integer, n_off integer)''')    
    excess.to_sql(name="run", con=conn, if_exists="append", index=False)
    conn.close()

#Sourcename from CSV
def sourcename(source_x, source_y, pointing_zd, pointing_az, timestamp):
    sourceList = camera_to_equatorial(
        source_x,
        source_y,
        pointing_zd,
        pointing_az,
        timestamp,
    )
    source = SkyCoord(15*sourceList[0], sourceList[1], unit="deg")
    df = pd.read_csv('fact_sources.csv')
    names = list(df.fSourceName)
    catalog = SkyCoord(
            ra=df.fRightAscension.values * u.hourangle,
            dec=df.fDeclination.values * u.deg,
            )
    idx, sep, dst = source.match_to_catalog_sky(catalog)
    return names[idx]

def to_altaz(obstime, source):
    altaz = AltAz(location=LOCATION, obstime=obstime)
    return source.transform_to(altaz) 

def concat_results_altaz(results):
    obstime = np.concatenate([s.obstime for s in results])
    return SkyCoord(
        alt=np.concatenate([s.alt.deg for s in results]) * u.deg,
        az=np.concatenate([s.az.deg for s in results]) * u.deg,
        frame=AltAz(location=LOCATION, obstime=obstime)
    )

def calc_source_features_common(
    source_x,
    source_y,
    source_zd,
    source_az,
    pointing_position_zd,
    pointing_position_az,
):
    result = {}
    result['theta_deg'] = calc_theta_camera(
        source_x,
        source_y,
        source_zd=source_zd,
        source_az=source_az,
        zd_pointing=pointing_position_zd,
        az_pointing=pointing_position_az,
    )
    theta_offs = calc_theta_offs_camera(
        source_x,
        source_y,
        source_zd=source_zd,
        source_az=source_az,
        zd_pointing=pointing_position_zd,
        az_pointing=pointing_position_az,
        n_off=5,
    )
    for i, theta_off in enumerate(theta_offs, start=1):
        result['theta_deg_off_{}'.format(i)] = theta_off
    return result


def calc_source_features_sim(
    source_x,
    source_y,
    source_zd,
    source_az,
    pointing_position_zd,
    pointing_position_az,
    cog_x,
    cog_y,
    delta,
):
    result = calc_source_features_common(
        source_x,
        source_y,
        source_zd,
        source_az,
        pointing_position_zd,
        pointing_position_az,
    )
    source_x, source_y = horizontal_to_camera(
        az=source_az,
        zd=source_zd,
        az_pointing=pointing_position_az,
        zd_pointing=pointing_position_zd,
    )

    true_disp, true_sign = calc_true_disp(
        source_x, source_y,
        cog_x, cog_y,
        delta,
    )
    result['true_disp'] = true_disp * true_sign
    return result


def calc_source_features_obs(
    source_x,
    source_y,
    source_zd,
    source_az,
    pointing_position_zd,
    pointing_position_az,
    obstime,
):
    result = calc_source_features_common(
        source_x,
        source_y,
        source_zd,
        source_az,
        pointing_position_zd,
        pointing_position_az,
    )
    return result

if __name__ == '__main__':
    loadfits()