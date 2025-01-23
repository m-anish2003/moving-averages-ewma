import pandas as pd
import numpy as np
%matplotlib inline

airline = pd.read_csv('../Data/airline_passengers.csv',index_col='Month',parse_dates=True)

airline.dropna(inplace=True)

airline.head()

airline['6-month-SMA'] = airline['Thousands of Passengers'].rolling(window=6).mean()
airline['12-month-SMA'] = airline['Thousands of Passengers'].rolling(window=12).mean()


airline.head(15)
airline.plot()



airline['EWMA12'] = airline['Thousands of Passengers'].ewm(span=12,adjust=False).mean()


airline[['Thousands of Passengers','EWMA12']].plot();

airline[['Thousands of Passengers','EWMA12','12-month-SMA']].plot(figsize=(12,8)).autoscale(axis='x',tight=True);