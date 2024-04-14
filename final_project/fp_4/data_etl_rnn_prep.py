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

def yf_data_upload(start_date, end_date, ticker_symbol):
    '''
    Call the yfinance and upload the required data. For our data requirements
    we will choose a start date of 2010-12-31 and an end date of 2024-02-29.

    Input:
        - start: string date in the format of yyyy-dd-mm for the start date of the data putll
        - end: string date in the format of yyyy-dd-mm for the end date of the data pull
        - ticker_symbol: string value in all caps that will be used for the ticker in our analysis
    
    return: DataFrame of historical stock price data
    '''
    # fetching daily adjusted close price, volume, and percent changes
    equity_price = yf.download(ticker_symbol, start=start_date, end=end_date, group_by='ticker')[['Adj Close', 'Volume']]
    equity_price[f'{ticker_symbol}_ret'] = equity_price['Adj Close'].pct_change()
    equity_price.rename(columns={'Adj Close': f'{ticker_symbol}_close', 'Volume': f'{ticker_symbol}_volume'}, inplace=True)

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
    df = pd.merge(equity_price, tbf_price, left_index=True, right_index=True)
    df = pd.merge(df, tip_price, left_index=True, right_index=True)
    df = pd.merge(df, uup_price, left_index=True, right_index=True)
    df = pd.merge(df, vixy_price, left_index=True, right_index=True)
    df = pd.merge(df, uso_price, left_index=True, right_index=True)
    df = pd.merge(df, gld_price, left_index=True, right_index=True)
    df.reset_index(inplace=True)

    return df


def feature_engineering(df, ticker_symbol):
    '''
    Creating required features that are fundamental in the investment process.

    Inputs:
        - DataFrame of historical stock price data.
        - ticker_symbol: string value in all caps that will be used for the ticker in our analysis

    Return: DataFrame containing additional engineered features.
    '''
    # feature engineering {ticker_symbol} technical indicator data
    # 35-day EMA
    df[f'{ticker_symbol}_35_day_ema'] = df[f'{ticker_symbol}_close'].ewm(span=35, adjust=False).mean()

    # Calculate 200-day SMA
    df[f'{ticker_symbol}_200_day_sma'] = df[f'{ticker_symbol}_close'].rolling(window=200).mean()

    # Calculate MACD and MACD signal
    df[f'{ticker_symbol}_macd'] = df[f'{ticker_symbol}_close'].ewm(span=12, adjust=False).mean() - df[f'{ticker_symbol}_close'].ewm(span=26, adjust=False).mean()
    df[f'{ticker_symbol}_macd_signal'] = df[f'{ticker_symbol}_macd'].ewm(span=9, adjust=False).mean()

    # Calculate RSI
    delta = df[f'{ticker_symbol}_close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df[f'{ticker_symbol}_rsi'] = 100 - (100 / (1 + rs))
    df.fillna(0, inplace=True)
    
    return df


def simulate_ret(df, ticker_symbol):
    '''
    Simulate the perfect investment strategy with already knowing future equity prices.

    Input:
        - df: DataFrame containing all stock price data.
        - ticker_symbol: string value in all caps that will be used for the ticker in our analysis

    Return: df DataFrame with the simulated returns and dependent variables.
    '''

    # buy, sell, hold decision variable
    df['decision'] = ''

    # simulating cash position set to the starting price of {ticker_symbol}
    df['cash'] = df[f'{ticker_symbol}_close'][0]

    # daily return on cash, significant for shorts and if we need to add cumulative return at any point
    df['cash_ret'] = df[f'{ticker_symbol}_ret'][0]

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
                    1: (df[f'{ticker_symbol}_close'][idx+1]/df[f'{ticker_symbol}_close'][idx])-1,
                    3: (df[f'{ticker_symbol}_close'][idx+3]/df[f'{ticker_symbol}_close'][idx])-1,
                    5: (df[f'{ticker_symbol}_close'][idx+5]/df[f'{ticker_symbol}_close'][idx])-1,
                    10: (df[f'{ticker_symbol}_close'][idx+10]/df[f'{ticker_symbol}_close'][idx])-1,
                    20: (df[f'{ticker_symbol}_close'][idx+20]/df[f'{ticker_symbol}_close'][idx])-1
                }

                # if all future values are negative, short the stock
                if all(value < 0 for value in hold_dict.values()):

                    # when the initial position is a short, finish the current day as short
                    if hold_type == 'short':
                        
                        df['decision'][idx] = 'short'
                        days_to_hold = min(hold_dict, key=hold_dict.get)
                        df['initial_hold_days'][idx] = days_to_hold
                        df['hold_countdown'][idx] = days_to_hold
                        df['cash'][idx] = df['cash'][idx-1]*(1+(df[f'{ticker_symbol}_ret'][idx]*-1))
                        df['cash_ret'][idx] = (df['cash'][idx]/df['cash'][idx-1])-1
                        hold_type = 'short'

                    # when the initial position is a long, finish the current day as long
                    elif hold_type == 'long':

                        df['decision'][idx] = 'short'
                        days_to_hold = min(hold_dict, key=hold_dict.get)
                        df['initial_hold_days'][idx] = days_to_hold
                        df['hold_countdown'][idx] = days_to_hold
                        df['cash'][idx] = df['cash'][idx-1]*(1+(df[f'{ticker_symbol}_ret'][idx]))
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
                        df['cash'][idx] = df['cash'][idx-1]*(1+(df[f'{ticker_symbol}_ret'][idx]*-1))
                        df['cash_ret'][idx] = (df['cash'][idx]/df['cash'][idx-1])-1
                        hold_type = 'long'
                    
                    elif hold_type == 'long':
                        # when the initial position is a long, finish the current day as long
                        df['decision'][idx] = 'long'
                        days_to_hold = max(hold_dict, key=hold_dict.get)
                        df['initial_hold_days'][idx] = days_to_hold
                        df['hold_countdown'][idx] = days_to_hold
                        df['cash'][idx] = df['cash'][idx-1]*(1+(df[f'{ticker_symbol}_ret'][idx]))
                        df['cash_ret'][idx] = (df['cash'][idx]/df['cash'][idx-1])-1
                        hold_type = 'long'

            elif days_to_hold > 1:

                # when a days_to_hold is decided - continue position until end of the hold days
                if hold_type == 'long':
                    df['decision'][idx] = 'hold'
                    df['cash'][idx] = df['cash'][idx-1]*(1+df[f'{ticker_symbol}_ret'][idx])
                    df['cash_ret'][idx] = (df['cash'][idx]/df['cash'][idx-1])-1
                    days_to_hold -= 1
                    df['hold_countdown'][idx] = days_to_hold
                    df['initial_hold_days'][idx] = df['initial_hold_days'][idx-1]
                    
                elif hold_type == 'short':
                    df['decision'][idx] = 'hold'
                    df['cash'][idx] = df['cash'][idx-1]*(1+(df[f'{ticker_symbol}_ret'][idx]*-1)) 
                    df['cash_ret'][idx] = (df['cash'][idx]/df['cash'][idx-1])-1
                    days_to_hold -= 1
                    df['hold_countdown'][idx] = days_to_hold
                    df['initial_hold_days'][idx] = df['initial_hold_days'][idx-1]

        else:
            # when we get to the end of our dataframe hold our current position (long/short)
            if hold_type == 'long':
                df['decision'][idx] = 'hold'
                df['cash'][idx] = df['cash'][idx-1]*(1+df[f'{ticker_symbol}_ret'][idx])
                df['cash_ret'][idx] = (df['cash'][idx]/df['cash'][idx-1])-1
                df['hold_countdown'][idx] = 0
                df['initial_hold_days'][idx] = df['initial_hold_days'][idx-1]
                
            elif hold_type == 'short':
                df['decision'][idx] = 'hold'
                df['cash'][idx] = df['cash'][idx-1]*(1+(df[f'{ticker_symbol}_ret'][idx]*-1))
                df['cash_ret'][idx] = (df['cash'][idx]/df['cash'][idx-1])-1
                df['hold_countdown'][idx] = 0
                df['initial_hold_days'][idx] = df['initial_hold_days'][idx-1]
    
    # create future day close features
    df[f'{ticker_symbol}_1_day_close'] = 0
    df[f'{ticker_symbol}_3_day_close'] = 0
    df[f'{ticker_symbol}_5_day_close'] = 0
    df[f'{ticker_symbol}_10_day_close'] = 0
    df[f'{ticker_symbol}_20_day_close'] = 0

    # document future price values
    for idx in range(len(df)):
        if idx <= len(df)-21:

            df[f'{ticker_symbol}_1_day_close'][idx] = df[f'{ticker_symbol}_close'][idx+1]
            df[f'{ticker_symbol}_3_day_close'][idx] = df[f'{ticker_symbol}_close'][idx+3]
            df[f'{ticker_symbol}_5_day_close'][idx] = df[f'{ticker_symbol}_close'][idx+5]
            df[f'{ticker_symbol}_10_day_close'][idx] = df[f'{ticker_symbol}_close'][idx+10]
            df[f'{ticker_symbol}_20_day_close'][idx] = df[f'{ticker_symbol}_close'][idx+20]
        
        else:
            df[f'{ticker_symbol}_1_day_close'][idx] = 0
            df[f'{ticker_symbol}_3_day_close'][idx] = 0
            df[f'{ticker_symbol}_5_day_close'][idx] = 0
            df[f'{ticker_symbol}_10_day_close'][idx] = 0
            df[f'{ticker_symbol}_20_day_close'][idx] = 0

    # will only return data that reports the beginning of the 200 day moving average and the last 20 days
    df = df.iloc[199:-20]
    df = df.reset_index(drop=True)

    return df


def execute_etl(start_date, end_date, ticker_symbol):
    '''
    Execute the data extraction and features engineering functions

    Input:
        - start: string date in the format of yyyy-dd-mm for the start date of the data putll
        - end: string date in the format of yyyy-dd-mm for the end date of the data pull
        - ticker_symbol: string value in all caps that will be used for the ticker in our analysis
    
    Return: DataFrame of historical stock price data 
    '''
    df = yf_data_upload(start_date, end_date, ticker_symbol)
    df = feature_engineering(df, ticker_symbol)
    df = simulate_ret(df, ticker_symbol)

    return df


def rolling_window_scaling_and_save(series, window):
    '''
    Function used to normalize data, using min-max scalling, based on 
    user defined rolling window.

    Input:
        - series: dataframe defined as the dependent or independent variables.
        - window: rolling window period

    Return: series containing the scaled data, series containing min values, series containing max values
    '''
    min_series = series.rolling(window=window, min_periods=window).min()
    max_series = series.rolling(window=window, min_periods=window).max()
    scaled_series = (series - min_series) / (max_series - min_series)
    
    return scaled_series, min_series, max_series


def apply_feature_scaling(window, x_rnn_train, y_rnn_train=0, training_set=False):
    '''
    Function used to apply the rolling_window_scaling_and_save function more intuitively
    to our dataframes.

    Input:
        - window: integer rolling window period
        - x_rnn_train: dataframe of the independent variables
        - y_rnn_train: dataframe of the dependent variables
        - training_set: boolean if True then we will apply scaling for x and y, if False, then only apply scaling to y.

    Return: four dataframes containing normalized features and the min-max values for each rolling period row
    '''
    if training_set == True:
        # new normalized placeholder and min/max value placeholder to recall from the normalization
        x_rnn_train_normalized = pd.DataFrame(index=x_rnn_train.index)
        x_rnn_train_min_max_values = pd.DataFrame(index=x_rnn_train.index)

        # y min_max values will be used to convert the normalized y back to the actual values
        y_rnn_train_normalized = pd.DataFrame(index=y_rnn_train.index)
        y_rnn_train_min_max_values = pd.DataFrame(index=y_rnn_train.index)

        # apply rolling window scaling to features and save min/max
        for feature in x_rnn_train.columns.values:
            x_rnn_train_normalized[f'{feature}_scaled'], x_rnn_train_min_max_values[f'{feature}_min'], x_rnn_train_min_max_values[f'{feature}_max'] = rolling_window_scaling_and_save(x_rnn_train[feature], window)

        for feature in y_rnn_train.columns.values:
            y_rnn_train_normalized[f'{feature}_scaled'], y_rnn_train_min_max_values[f'{feature}_min'], y_rnn_train_min_max_values[f'{feature}_max'] = rolling_window_scaling_and_save(y_rnn_train[feature], window)

        x_rnn_train_normalized.dropna(inplace=True)
        y_rnn_train_normalized.dropna(inplace=True)
        x_rnn_train_min_max_values.dropna(inplace=True)
        y_rnn_train_min_max_values.dropna(inplace=True)

        return x_rnn_train_normalized, x_rnn_train_min_max_values, y_rnn_train_normalized, y_rnn_train_min_max_values
    
    else:
        # new normalized placeholder and min/max value placeholder to recall from the normalization
        x_rnn_train_normalized = pd.DataFrame(index=x_rnn_train.index)
        x_rnn_train_min_max_values = pd.DataFrame(index=x_rnn_train.index)

        # apply rolling window scaling to features and save min/max
        for feature in x_rnn_train.columns.values:
            x_rnn_train_normalized[f'{feature}_scaled'], x_rnn_train_min_max_values[f'{feature}_min'], x_rnn_train_min_max_values[f'{feature}_max'] = rolling_window_scaling_and_save(x_rnn_train[feature], window)

        x_rnn_train_normalized.dropna(inplace=True)
        x_rnn_train_min_max_values.dropna(inplace=True)

        return x_rnn_train_normalized, x_rnn_train_min_max_values


def reverse_rolling_window_scaling(scaled_series, min_series, max_series):
    '''
    Function used to reverse the normalization of data, using min-max values
    which is created primarily to interperate the results of predictions.

    Input:
        - scaled_series: series 
        - min_series: series of min original values for scaled series
        - max_series: series of max original values for scaled series

    Return: series containing the scaled data, series containing min values, series containing max values
    '''
    original_series = scaled_series * (max_series - min_series) + min_series
    
    return original_series


def re_shape_df(sequence_length, x_rnn_train_normalized, y_rnn_train_normalized=0, train_set=False):
    '''
    Reshapping the independent and dependent variable data to run in our RNN. If train_set is equal to False then
    that assumes that this re-shapping will be applied to make a prediction.

    Input:
        - sequence_length: integer value that represents the number of days which the RNN model will consider
        - x_rnn_train_normalized: dataframe containing our independent variables
        - y_rnn_train_normalized: dataframe containing our dependent varialbes
        - train_set: boolean indicating if re-shapping will be applied to a test set or to make predictions

    Return: reshapped X and y data
    '''

    # convert dataframe to np array if not already (for safe indexing)
    x_rnn_train_normalized = x_rnn_train_normalized.to_numpy() if hasattr(x_rnn_train_normalized, 'to_numpy') else x_rnn_train_normalized
    y_rnn_train_normalized = y_rnn_train_normalized.to_numpy() if hasattr(y_rnn_train_normalized, 'to_numpy') else y_rnn_train_normalized

    if train_set == True:
        # initialize the resulting X and y arrays for training the RNN
        X = np.zeros((len(x_rnn_train_normalized) - sequence_length, sequence_length, x_rnn_train_normalized.shape[1]))
        y = np.zeros((len(y_rnn_train_normalized) - sequence_length, y_rnn_train_normalized.shape[1]))

        for i in range(len(x_rnn_train_normalized) - sequence_length):
            X[i] = x_rnn_train_normalized[i:i+sequence_length]  # Shape [sequence_length, 26]
            y[i] = y_rnn_train_normalized[i + sequence_length - 1]  # Shape [5]

        print(f'shape of x: {X.shape}')
        print(f'shape of y: {y.shape}')

        return X, y

    else:
        # initialize the resulting X array for testing the RNN
        X = np.zeros((len(x_rnn_train_normalized) - sequence_length, sequence_length, len(x_rnn_train_normalized.columns)))

        for i in range(len(x_rnn_train_normalized) - sequence_length):
            X[i] = x_rnn_train_normalized[i:i+sequence_length]  # Shape [sequence_length, 26]

        print(f'shape of x train: {X.shape}')

        return X
