import keras
import time
import datetime
import plotnine as pn
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import numpy as np
import os
from twilio.rest import Client
import json


class Detector(object):

    buffer = []
    looker = "scanning...."
    client = Client("AC539802fc1540422060a13476ebfb7945", "42e2ed1ae4b75c844265268a3d488063")
    warnState = False

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

    def getRecs(self):
        delta = datetime.timedelta(milliseconds=32)
        time_format = '%Y-%m-%d %H:%M:%S'
        start = time.strptime("2019-04-22 20:15:00.000", time_format)
        records = [[], [start]]

        file = open("test_quake3.jsonl")
        for line in file:
            data = json.loads(line)
            for val in data["x"]:
                start += delta
                records[0].append(val)
                records[1].append(start)

        return pd.DataFrame(records, columns=["sample_dt", "x"])

    def plot_seismograms(self, device_id, records):
        # Get earthquake date as datetime.datetime object

        ob = {
            "ti": "2019-04-22 20:15:00"
        }
        time_format = '%Y-%m-%d %H:%M:%S'
        plots = []
        plots.append(
            pn.ggplot(
                records,
                pn.aes('sample_dt', 'x')
            ) + \
            pn.geom_line(color='blue') + \
            pn.scales.scale_x_datetime(
                date_breaks='1 minute',
                date_labels='%H:%M:%S'
            ) + \
            pn.geoms.geom_vline(
                xintercept=time.strptime(ob["ti"], time_format),
                color='crimson'
            ) + \
            pn.labels.ggtitle(
                'device {}, axis {}'.format(
                    device_id, 'x')
            )
        )
        # Now output the plots
        for p in plots:
            print(p)

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
            shuffle=False,
            use_multiprocessing=False
        )

    def process(self, model):

        testFile = open("./test_quake3.jsonl", "r")
        start_time = time.time()
        #records = self.getRecs()

        # Need to check if buffer is updated
        print("Start Processing")
        for obj in testFile:
            yArr, zArr, xArr = [], [], []
            lineData = json.loads(obj)
            xArr.append(lineData["x"])
            yArr.append(lineData["y"])
            zArr.append(lineData["z"])

            data = [xArr, yArr, zArr]

            if len(self.buffer) >= 19:
                self.buffer.pop(0)
                self.buffer.append(data)
            else:
                self.buffer.append(data)

            if len(self.buffer) >= 10:

                formattedData = []

                # transpose data format
                for ar in self.buffer:
                    for x in range(0, len(ar)):
                        for y in range(0, len(ar[x])):
                            for z in range(0, len(ar[x][y])):
                                formattedData.append([ar[0][y][z], ar[1][y][z], ar[2][y][z]])

                # insert into dataframe
                df = pd.DataFrame(data=formattedData, columns=["x", "y", "z"])

                # Start Pre Processing
                ar = []
                TIME_STEPS = 300
                THRESHOLD = 1.0

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

                print(self.looker)

                P_test, py_test = self.create_dataset(
                    p_test[['Movement']],
                    p_test.Movement,
                    TIME_STEPS
                )
                run_time = time.time()
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

                if not anomaliesPwave.empty and not self.warnState:
                    self.client.messages.create(to="+12543156540",
                                                from_="+12816889321",
                                                body="WARNING: POSSIBLE EARTHQUAKE INCOMING")
                    print("EARTHQUAKE DETECTED AT: " + str(time.time() - start_time))

                    self.looker = "QUAKE DETECTED, STILL LOOKING"
                    self.warnState = True
                    plt.scatter(x=score_df.index, y=score_df.loss, c=score_df.Movement, s=2,
                                cmap="viridis")
                    plt.colorbar()
                    plt.show()
                    plt.close()

                    if anomaliesPwave.index[0]:
                        plt.axvline(x=anomaliesPwave.index[0], c="crimson")

                    # plt.axvline(x=score_df.index[lastIndex], c="crimson")

                    loss_values = score_df['loss']
                    epochs = range(1, len(loss_values) + 1)

                    plt.plot(epochs, loss_values, label='Value Loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.legend()

                    plt.show()
                    aLoss = sum(loss_values)/len(epochs)
                    stdLoss = statistics.stdev(loss_values)

                    print("Average Loss: " + str(aLoss))
                    print("STD of loss: " + str(stdLoss))


                elif anomaliesPwave.empty and self.warnState:
                    self.warnState = False
                    self.looker = "scanning...."



def main():
    detector = Detector()

    if "model" not in os.listdir("./"):
        train, test = detector.setTrainData()
        X_train, y_train = detector.dataPreProcessing(train, test)
        model = detector.generateModel(X_train)
        detector.trainModel(X_train, y_train, model)
        model.save("./model")
        print("Generating model")
    else:
        print("Model imported")
        model = keras.models.load_model('./model')

    detector.process(model)



if __name__ == "__main__":
    main()
