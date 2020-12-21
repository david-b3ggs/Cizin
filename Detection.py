import keras
import time
import threading
from keras.models import Model
from keras import Sequential
from keras.layers import LSTM, Dropout, RepeatVector
from keras.utils import plot_model
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
import pandas as pd
import numpy as np

from EarlyWarning import WarningSystem

class Detector(object):

    data = []

    def setData(self, data):
        self.data = data

    def create_dataset(self, X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)

    def setTrainData(self):
        df = pd.read_csv('./output.csv', header=None)
        print(df.head())

        train_size = int(len(df) * 0.7)
        test_size = len(df) - train_size

        origin = (0, 0, 0)

        # define input sequence

        # first take 3 dimension arrays, run avg of abs for every element and produce
        # input sequence. Taking distance form is harder to learn
        seq = []

        # print(df.values[0])
        # norm = normalize(df.values[1:], norm='l2')
        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(df.values[1:])
        fixed = min_max_scaler.transform(df.values[1:])

        for point in fixed:
            seq.append((point[0] + point[1] + point[2]) / 3)

        fram = pd.DataFrame(data=seq, columns=["Movement"])
        print(fram.head())

        train, test = fram.iloc[0:train_size], fram.iloc[train_size:len(df)]

        print(train.shape, test.shape)

        return train, test

    def dataPreProcessing(self, train, test):
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        scaler = StandardScaler()
        scaler = scaler.fit(train[['Movement']])
        train['Movement'] = scaler.transform(train[['Movement']])
        test['Movement'] = scaler.transform(test[['Movement']])

        def create_dataset(X, y, time_steps=1):
            Xs, ys = [], []
            for i in range(len(X) - time_steps):
                v = X.iloc[i:(i + time_steps)].values
                Xs.append(v)
                ys.append(y.iloc[i + time_steps])
            return np.array(Xs), np.array(ys)

        # Reference last 10 seconds
        TIME_STEPS = 300
        # reshape to [samples, time_steps, n_features]
        X_train, y_train = create_dataset(
            train[['Movement']],
            train.Movement,
            TIME_STEPS
        )
        X_test, y_test = create_dataset(
            test[['Movement']],
            test.Movement,
            TIME_STEPS
        )

        return X_train, y_train

    def generateModel(self, X_train):
        model = keras.Sequential()
        model.add(keras.layers.LSTM(
            units=32,
            input_shape=(X_train.shape[1], X_train.shape[2])
        ))
        model.add(keras.layers.Dropout(rate=0.7))
        model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
        model.add(keras.layers.LSTM(units=32, return_sequences=True))
        model.add(keras.layers.Dropout(rate=0.7))
        model.add(
            keras.layers.TimeDistributed(
                keras.layers.Dense(units=X_train.shape[2])
            )
        )
        model.compile(loss='mae', optimizer='adagrad')
        return model

    def trainModel(self, X_train, y_train, model):
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.1,
            shuffle=False
        )


    def process(self, model, buffer, warner):

        # Need to check if buffer is updated
        while True:

            status = buffer.getStatus()
            status.wait()
            # start_time = time.time()
            data = buffer.read(self)
            formattedData = []

            # transpose data format

            for x in range(0, len(data[0])):
                formattedData.append([data[0][x], data[1][x], data[2][x]])

            # insert into dataframe
            df = pd.DataFrame(data=formattedData, columns=["x", "y", "z"])

            testLen = len(df)

            # Start Pre Processing
            ar = []
            TIME_STEPS = 300
            THRESHOLD = 0.9

            min_max_scaler = MinMaxScaler()
            min_max_scaler.fit(df.values[1:])
            tr = min_max_scaler.transform(df.values[1:])

            for point in tr:
                ar.append((point[0] + point[1] + point[2]) / 3)

            pFrame = pd.DataFrame(data=ar, columns=["Movement"])
            p_test = pFrame.iloc[0:len(df)]

            scaler = StandardScaler()
            scaler = scaler.fit(p_test[['Movement']])
            p_test['Movement'] = scaler.transform(p_test[['Movement']])

            P_test, py_test = self.create_dataset(
                p_test[['Movement']],
                p_test.Movement,
                TIME_STEPS
            )

            # Run the Model on data set
            P_test_pred = model.predict(P_test)
            more_mae_loss = np.mean(np.abs(P_test_pred - P_test), axis=1)

            # push into data frame and detect anomolous movement
            score_df = pd.DataFrame(index=p_test[TIME_STEPS:].index)
            score_df['loss'] = more_mae_loss
            score_df['threshold'] = THRESHOLD
            score_df['anomaly'] = score_df.loss > score_df.threshold
            score_df['Movement'] = p_test[TIME_STEPS:].Movement

            anomaliesPwave = score_df[score_df.anomaly == True]

            if anomaliesPwave:
                # Set warning state
                warner.setWarning(True)
            else:
                warner.setWarning(False)

