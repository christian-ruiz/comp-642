'''
This file contains all necessary code for conducting a simulation
on returns following the predictions using the developed RNN on 
TQQQ ETF data. STILL NEED TO CODE THIS SOLUTION OUT.

Author: Christian Ruiz, cr72@rice.edu
Course: COMP 642
Instructor: Janell Straach
'''


def simulate_ret(df):
    '''
    Using predicted prices to simulate possible returns.

    Input:
        - df: DataFrame holding the predicted future price data.
    
    return: DataFrame of simulated returns
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

