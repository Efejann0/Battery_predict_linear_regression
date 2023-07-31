from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import sendemail as  sendemail

def battery_predict(data):
    report = """ <!DOCTYPE html><html><head>    <style>
        .styled-table {{
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.9em;
            font-family: sans-serif;
            min-width: 400px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        }}
        .styled-table thead tr {{
            background-color: #009879;
            color: #ffffff;
            text-align: left;
        }}
        .styled-table th,
        .styled-table td {{
            padding: 12px 15px;
        }}
        .styled-table tbody tr {{
            border-bottom: 1px solid #dddddd;
        }}
        .styled-table tbody tr:nth-of-type(even) {{
            background-color: #f3f3f3;
        }}
        .styled-table tbody tr:last-of-type {{
            border-bottom: 2px solid #009879;
        }}
    </style></head><body>
    <div class="container">
        <table style="width:100%" class="styled-table">
            <thead>
                            <tr>
                            <th>{}</th>
                            <th>{}</th>
                            </tr></thead><body>""".format('Yabby','Kalan Gün')
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
            # tscv = TimeSeriesSplit(n_splits=5)

            # Evaluate the model using MSE
            # mse = make_scorer(mean_squared_error)
            # scores = cross_val_score(regressor, X, y, cv=tscv, scoring=mse)

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
                    # print("Prediction for Yabby {} cannot exceed 20 years.".format(yabby))
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

            # If the battery life is 10 days or less, give a warning message
            if gun <= 10:
                # print("Attention! The battery of Yabby {} will be 0 within {} days.".format(yabby, gun))
                report += """
                                            <tr>
                                            <td >{}</td>
                                            <td >{} günlük batarya ömrü kalmıştır.</td>
                                            </tr>
                                            """.format(yabby,gun)
                
                plt.plot(dates, predictions, label="{}".format(yabby))
            else:
                pass
                # print("The battery of Yabby {} will be 0 in {} days.".format(yabby,gun))

            # Plotting
            # plt.plot(dates, predictions, label="{}".format(yabby))

        else:
            pass
            # print("There is not enough data for Yabby {}.".format(yabby))

    report += """ </tbody></table></div></body></html>"""
    # Setting the title, labels, and legend
    plt.title('Battery Predictions for all Yabbies')
    plt.xlabel('Date')
    plt.ylabel('Battery Life')
    plt.legend()
    plt.savefig('yabby_predictions.png')
    # plt.show()
    sendemail.send_mail(report, 'yabby_predictions.png')


# ------------------------ TEKER TEKER GRAFIK ---------------------------------------
# from sklearn.model_selection import TimeSeriesSplit, cross_val_score
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, make_scorer
# import pandas as pd
# import numpy as np
# import datetime
# import matplotlib.pyplot as plt
# import sendemail as  sendemail

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
#                 email = ("Attention! The battery of Yabby {} will be 0 within {} days.".format(yabby, gun))
#                 sendemail.send_mail(email)
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
