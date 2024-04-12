"""
This file contains all necessary code for extracting 
the required data for COMP 642 final project, Spring 2024.

Author: Christian Ruiz, cr72@rice.edu
Course: COMP 642
Instructor: Janell Straach
"""

# import necessary packages
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def yf_data_upload(start_date, end_date):
    '''
    Call the yfinance and upload the required data. For our data requirements
    we will choose a start date of 2010-12-31 and an end date of 2024-02-29.

    Input:
        - start: string date in the format of yyyy-dd-mm for the start date of the data putll
        - end: string date in the format of yyyy-dd-mm for the end date of the data pull
    
    return: DataFrame of historical stock price data
    '''
    # fetching daily adjusted close price, volume, and percent changes
    tqqq_price = yf.download('TQQQ', start=start_date, end=end_date, group_by='ticker')[['Adj Close', 'Volume']]
    tqqq_price['tqqq_ret'] = tqqq_price['Adj Close'].pct_change()
    tqqq_price.rename(columns={'Adj Close': 'tqqq_close', 'Volume': 'tqqq_volume'}, inplace=True)

    tbf_price = pd.DataFrame(yf.download('TBF', start=start_date, end=end_date)[['Adj Close', 'Volume']])
    tbf_price['tbf_ret'] = tbf_price['Adj Close'].pct_change()
    tbf_price.rename(columns={'Adj Close': 'tbf_close', 'Volume': 'tbf_volume'}, inplace=True)

    tip_price = pd.DataFrame(yf.download('TIP', start=start_date, end=end_date)[['Adj Close', 'Volume']])
    tip_price['tip_ret'] = tip_price['Adj Close'].pct_change()
    tip_price.rename(columns={'Adj Close': 'tip_close', 'Volume': 'tip_volume'}, inplace=True)

    uup_price = pd.DataFrame(yf.download('UUP', start=start_date, end=end_date)[['Adj Close', 'Volume']])
    uup_price['uup_ret'] = uup_price['Adj Close'].pct_change()
    uup_price.rename(columns={'Adj Close': 'uup_close', 'Volume': 'uup_volume'}, inplace=True)

    vixy_price = pd.DataFrame(yf.download('VIXY', start=start_date, end=end_date)[['Adj Close', 'Volume']])
    vixy_price['vixy_ret'] = vixy_price['Adj Close'].pct_change()
    vixy_price.rename(columns={'Adj Close': 'vixy_close', 'Volume': 'vixy_volume'}, inplace=True)

    uso_price = pd.DataFrame(yf.download('USO', start=start_date, end=end_date)[['Adj Close', 'Volume']])
    uso_price['uso_ret'] = uso_price['Adj Close'].pct_change()
    uso_price.rename(columns={'Adj Close': 'uso_close', 'Volume': 'uso_volume'}, inplace=True)

    gld_price = pd.DataFrame(yf.download('GLD', start=start_date, end=end_date)[['Adj Close', 'Volume']])
    gld_price['gld_ret'] = gld_price['Adj Close'].pct_change()
    gld_price.rename(columns={'Adj Close': 'gld_close', 'Volume': 'gld_volume'}, inplace=True)

    # joining all data into a single dataframe according to the index (date)
    df = pd.merge(tqqq_price, tbf_price, left_index=True, right_index=True)
    df = pd.merge(df, tip_price, left_index=True, right_index=True)
    df = pd.merge(df, uup_price, left_index=True, right_index=True)
    df = pd.merge(df, vixy_price, left_index=True, right_index=True)
    df = pd.merge(df, uso_price, left_index=True, right_index=True)
    df = pd.merge(df, gld_price, left_index=True, right_index=True)
    df.reset_index(inplace=True)

    return df


def feature_engineering(df):
    '''
    Creating required features that are fundamental in the investment process.

    Inputs:
        - DataFrame of historical stock price data.

    return: DataFrame containing additional engineered features.
    '''
    # feature engineering tqqq technical indicator data
    # 35-day EMA
    df['tqqq_35_day_ema'] = df['tqqq_close'].ewm(span=35, adjust=False).mean()

    # Calculate 200-day SMA
    df['tqqq_200_day_sma'] = df['tqqq_close'].rolling(window=200).mean()

    # Calculate MACD and MACD signal
    df['tqqq_macd'] = df['tqqq_close'].ewm(span=12, adjust=False).mean() - df['tqqq_close'].ewm(span=26, adjust=False).mean()
    df['tqqq_macd_signal'] = df['tqqq_macd'].ewm(span=9, adjust=False).mean()

    # Calculate RSI
    delta = df['tqqq_close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['tqqq_rsi'] = 100 - (100 / (1 + rs))
    df.fillna(0, inplace=True)
    
    return df


def simulate_ret(df):
    '''
    Simulate the perfect investment strategy with already knowing future equity prices.

    Input:
        - df: DataFrame containing all stock price data.

    return df DataFrame with the simulated returns and dependent variables.
    '''

    # buy, sell, hold decision variable
    df['decision'] = ''

    # simulating cash position set to the starting price of tqqq
    df['cash'] = df['tqqq_close'][0]

    # daily return on cash, significant for shorts and if we need to add cumulative return at any point
    df['cash_ret'] = df['tqqq_ret'][0]

    # the initial number of days that we are set to hold a security, could be used as categorical at any point
    df['initial_hold_days'] = 1

    # countdown check to be sure code is executing properly
    df['hold_countdown'] = 1

    # variable we will use as a countdown indicator for updating the iterative algorithm
    days_to_hold = 0

    # indicating how we will change the return value, either maintaining positive for buy or inverse for shorts
    hold_type = 'long'

    for idx in range(len(df)):
        if idx == 0:
            # decision day 0 as long as entry into the market
            df['decision'][idx] = 'long'

        elif (idx < len(df)-20) & (idx != 0):

            # reaching the end of the days_to_hold, trigger new hold_dict with future returns
            if days_to_hold <= 1:
            
                # determine how many days to hold for greatest return
                hold_dict = {
                    1: (df['tqqq_close'][idx+1]/df['tqqq_close'][idx])-1,
                    3: (df['tqqq_close'][idx+3]/df['tqqq_close'][idx])-1,
                    5: (df['tqqq_close'][idx+5]/df['tqqq_close'][idx])-1,
                    10: (df['tqqq_close'][idx+10]/df['tqqq_close'][idx])-1,
                    20: (df['tqqq_close'][idx+20]/df['tqqq_close'][idx])-1
                }

                # if all future values are negative, short the stock
                if all(value < 0 for value in hold_dict.values()):

                    # when the initial position is a short, finish the current day as short
                    if hold_type == 'short':
                        
                        df['decision'][idx] = 'short'
                        days_to_hold = min(hold_dict, key=hold_dict.get)
                        df['initial_hold_days'][idx] = days_to_hold
                        df['hold_countdown'][idx] = days_to_hold
                        df['cash'][idx] = df['cash'][idx-1]*(1+(df['tqqq_ret'][idx]*-1))
                        df['cash_ret'][idx] = (df['cash'][idx]/df['cash'][idx-1])-1
                        hold_type = 'short'

                    # when the initial position is a long, finish the current day as long
                    elif hold_type == 'long':

                        df['decision'][idx] = 'short'
                        days_to_hold = min(hold_dict, key=hold_dict.get)
                        df['initial_hold_days'][idx] = days_to_hold
                        df['hold_countdown'][idx] = days_to_hold
                        df['cash'][idx] = df['cash'][idx-1]*(1+(df['tqqq_ret'][idx]))
                        df['cash_ret'][idx] = (df['cash'][idx]/df['cash'][idx-1])-1
                        hold_type = 'short'

                else:

                    # go long otherwise
                    if hold_type == 'short':
                        # when the initial position is a short, finish the current day as short
                        df['decision'][idx] = 'long'
                        days_to_hold = max(hold_dict, key=hold_dict.get)
                        df['initial_hold_days'][idx] = days_to_hold
                        df['hold_countdown'][idx] = days_to_hold
                        df['cash'][idx] = df['cash'][idx-1]*(1+(df['tqqq_ret'][idx]*-1))
                        df['cash_ret'][idx] = (df['cash'][idx]/df['cash'][idx-1])-1
                        hold_type = 'long'
                    
                    elif hold_type == 'long':
                        # when the initial position is a long, finish the current day as long
                        df['decision'][idx] = 'long'
                        days_to_hold = max(hold_dict, key=hold_dict.get)
                        df['initial_hold_days'][idx] = days_to_hold
                        df['hold_countdown'][idx] = days_to_hold
                        df['cash'][idx] = df['cash'][idx-1]*(1+(df['tqqq_ret'][idx]))
                        df['cash_ret'][idx] = (df['cash'][idx]/df['cash'][idx-1])-1
                        hold_type = 'long'

            elif days_to_hold > 1:

                # when a days_to_hold is decided - continue position until end of the hold days
                if hold_type == 'long':
                    df['decision'][idx] = 'hold'
                    df['cash'][idx] = df['cash'][idx-1]*(1+df['tqqq_ret'][idx])
                    df['cash_ret'][idx] = (df['cash'][idx]/df['cash'][idx-1])-1
                    days_to_hold -= 1
                    df['hold_countdown'][idx] = days_to_hold
                    df['initial_hold_days'][idx] = df['initial_hold_days'][idx-1]
                    
                elif hold_type == 'short':
                    df['decision'][idx] = 'hold'
                    df['cash'][idx] = df['cash'][idx-1]*(1+(df['tqqq_ret'][idx]*-1)) 
                    df['cash_ret'][idx] = (df['cash'][idx]/df['cash'][idx-1])-1
                    days_to_hold -= 1
                    df['hold_countdown'][idx] = days_to_hold
                    df['initial_hold_days'][idx] = df['initial_hold_days'][idx-1]

        else:
            # when we get to the end of our dataframe hold our current position (long/short)
            if hold_type == 'long':
                df['decision'][idx] = 'hold'
                df['cash'][idx] = df['cash'][idx-1]*(1+df['tqqq_ret'][idx])
                df['cash_ret'][idx] = (df['cash'][idx]/df['cash'][idx-1])-1
                df['hold_countdown'][idx] = 0
                df['initial_hold_days'][idx] = df['initial_hold_days'][idx-1]
                
            elif hold_type == 'short':
                df['decision'][idx] = 'hold'
                df['cash'][idx] = df['cash'][idx-1]*(1+(df['tqqq_ret'][idx]*-1))
                df['cash_ret'][idx] = (df['cash'][idx]/df['cash'][idx-1])-1
                df['hold_countdown'][idx] = 0
                df['initial_hold_days'][idx] = df['initial_hold_days'][idx-1]




def execute_etl(start_date, end_date):
    '''
    Execute the data extraction and features engineering functions

    Input:
        - start: string date in the format of yyyy-dd-mm for the start date of the data putll
        - end: string date in the format of yyyy-dd-mm for the end date of the data pull
    
    return: DataFrame of historical stock price data 
    '''
    df = yf_data_upload(start_date, end_date)
    df = feature_engineering(df)
    df = simulate_ret(df)

    return df