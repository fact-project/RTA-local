# Real-Time Analysis (for local server)

Starting with Level 2 Data and sending results of analysis to database on web server.

## Analysis
Python script loads .fit files from ./Data/ directory and executes ML-Models for "Energy-Prediction", "Gamma-Prediction" and "Disp-Prediction". Performing Theta-Cut and writing results in SQL-Database.
After Analysis .fit files are redirected to ./Data/done/ to ensure File is just read once. 


For handling DB on web server use:
[RTA-web](https://github.com/fact-project/RTA-web)
