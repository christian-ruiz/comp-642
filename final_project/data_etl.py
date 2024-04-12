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
import matplotlib.pyplot as plt
import seaborn as sns
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

    return df