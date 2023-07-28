from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

def battery_predict(data):
    # Load data into pandas dataframe
    df_all_yabby = pd.DataFrame(data)
    yabbys = list(set(df_all_yabby['yabby_kod']))

    plt.figure(figsize=(12, 6))

    for yabby in yabbys:
        df = df_all_yabby[df_all_yabby['yabby_kod'] == yabby].copy()

        # Convert date information to numerical form (as epoch time)
        df['datalogged'] = pd.to_datetime(df['datalogged'])
        df['datalogged'] = (df['datalogged'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

        # Check for battery changes
        df['battery_diff'] = df['battery'].diff()
        battery_change_indices = df[df['battery_diff'] > 100].index

        if not battery_change_indices.empty:
            latest_battery_change_index = battery_change_indices[-1]
            df = df.loc[latest_battery_change_index:]

        # Define independent variables (X) and dependent variable (y)
        X = df['datalogged'].values.reshape(-1,1)
        y = df['battery'].values

        # If the dataset is large enough
        if len(X) > 1:
            # Train the Linear Regression model
            regressor = LinearRegression()

            # Apply Time-Series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)

            # Evaluate the model using MSE
            mse = make_scorer(mean_squared_error)
            scores = cross_val_score(regressor, X, y, cv=tscv, scoring=mse)

            # print("MSE scores for Yabby {}: {}".format(yabby, scores))

            regressor.fit(X, y)  # Train the model with the entire dataset

            # Starting date
            baslangic_tarihi = datetime.datetime.now()

            # Prepare arrays to store predictions and dates for plotting
            predictions = []
            dates = []
            gun = 0
            while True:
                if gun > 365*20:  # 20 years
                    print("Prediction for Yabby {} cannot exceed 20 years.".format(yabby))
                    break

                tarih = baslangic_tarihi + datetime.timedelta(days=gun)
                tarih_saniye = (tarih - datetime.datetime(1970, 1, 1)) // datetime.timedelta(seconds=1)
                batarya_suresi = regressor.predict(np.array([[tarih_saniye]]))

                predictions.append(batarya_suresi[0])
                dates.append(tarih)

                if batarya_suresi <= 2500:  # Find the date where y is 2500
                    break

                # Update the values for the next day
                gun += 1

            # If the battery life is 7 days or less, give a warning message
            if gun <= 7:
                print("Attention! The battery of Yabby {} will be 0 within {} days.".format(yabby, gun))
            else:
                print("The battery of Yabby {} will be 0 in {} days.".format(yabby,gun))

            # Plotting
            plt.plot(dates, predictions, label="{}".format(yabby))

        else:
            print("There is not enough data for Yabby {}.".format(yabby))
    
    # Setting the title, labels, and legend
    plt.title('Battery Predictions for all Yabbies')
    plt.xlabel('Date')
    plt.ylabel('Battery Life')
    plt.legend()
    plt.show()


# ------------------------ TEKER TEKER GRAFIK ---------------------------------------
# from sklearn.model_selection import TimeSeriesSplit, cross_val_score
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, make_scorer
# import pandas as pd
# import numpy as np
# import datetime
# import matplotlib.pyplot as plt

# def battery_predict(data):
#     # Load data into pandas dataframe
#     df_all_yabby = pd.DataFrame(data)
#     yabbys = list(set(df_all_yabby['yabby_kod']))

#     for yabby in yabbys:
#         df = df_all_yabby[df_all_yabby['yabby_kod'] == yabby].copy()

#         # Convert date information to numerical form (as epoch time)
#         df['datalogged'] = pd.to_datetime(df['datalogged'])
#         df['datalogged'] = (df['datalogged'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

#         # Check for battery changes
#         df['battery_diff'] = df['battery'].diff()
#         battery_change_indices = df[df['battery_diff'] > 100].index

#         if not battery_change_indices.empty:
#             latest_battery_change_index = battery_change_indices[-1]
#             df = df.loc[latest_battery_change_index:]

#         # Define independent variables (X) and dependent variable (y)
#         X = df['datalogged'].values.reshape(-1,1)
#         y = df['battery'].values

#         # If the dataset is large enough
#         if len(X) > 1:
#             # Train the Linear Regression model
#             regressor = LinearRegression()

#             # Apply Time-Series cross-validation
#             tscv = TimeSeriesSplit(n_splits=5)

#             # Evaluate the model using MSE
#             mse = make_scorer(mean_squared_error)
#             scores = cross_val_score(regressor, X, y, cv=tscv, scoring=mse)

#             regressor.fit(X, y)  # Train the model with the entire dataset

#             # Starting date
#             baslangic_tarihi = datetime.datetime.now()

#             # Prepare arrays to store predictions and dates for plotting
#             predictions = []
#             dates = []
#             gun = 0
#             while True:
#                 if gun > 365*20:  # 20 years
#                     print("Prediction for Yabby {} cannot exceed 20 years.".format(yabby))
#                     break

#                 tarih = baslangic_tarihi + datetime.timedelta(days=gun)
#                 tarih_saniye = (tarih - datetime.datetime(1970, 1, 1)) // datetime.timedelta(seconds=1)
#                 batarya_suresi = regressor.predict(np.array([[tarih_saniye]]))

#                 predictions.append(batarya_suresi[0])
#                 dates.append(tarih)

#                 if batarya_suresi <= 2500:  # Find the date where y is 2500
#                     break

#                 # Update the values for the next day
#                 gun += 1

#             # If the battery life is 7 days or less, give a warning message
#             if gun <= 7:
#                 print("Attention! The battery of Yabby {} will be 0 within {} days.".format(yabby, gun))
#             else:
#                 print("The battery of Yabby {} will be 0 in {} days.".format(yabby,gun))

#             # Plotting
#             plt.figure(figsize=(12, 6))
#             plt.plot(dates, predictions, label="Predictions for Yabby {}".format(yabby))
#             plt.title('Battery Predictions')
#             plt.xlabel('Date')
#             plt.ylabel('Battery Life')
#             plt.legend()
#             plt.show()

#         else:
#             print("There is not enough data for Yabby {}.".format(yabby))
