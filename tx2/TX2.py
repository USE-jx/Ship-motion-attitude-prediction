import pandas as pd
import numpy as np
import serial
import tensorflow as tf
import time
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential,load_model
import re
# from cnnlstmtpa import CalculateScoreMatrix
from sklearn.metrics import mean_absolute_percentage_error#MAPE
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import logging
import tensorflow as tf
import tensorflow.keras.backend as K
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Flatten, Dropout
from tensorflow.keras import layers, Input, optimizers, Model, losses, regularizers
from tensorflow.python.keras.metrics import accuracy
from sklearn.preprocessing import MinMaxScaler
from tempfile import TemporaryDirectory
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv1D, Reshape, Lambda, Flatten, Bidirectional, Layer, Dense, LSTM, \
    Activation, Multiply, Add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

ser = serial.Serial(
    port='/dev/ttyTHS2',
    baudrate=115200,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS
)
with tf.device('/cpu:0'):  # 使用CPU训练，本程序在GPU上训练的速度不如CPU，可能是模型深度与宽度不够的关系

    class DataLoader():
        def __init__(self, n_feature=1, lookback=9):
            self.n_feature = n_feature
            self.lookback = lookback

        def data_set(self, dataset):
            X, y = list(), list()
            for i in range(len(dataset)):
                end_ix = i + self.lookback
                if end_ix > len(dataset) - 1:
                    break
                seq_x, seq_y = dataset[i:end_ix, :], dataset[end_ix, :]
                X.append(seq_x)
                y.append(seq_y)
            return np.array(X), np.array(y)

        def spilt_data(self, dataset):
            train_size = int(len(dataset) * 0.8)
            test_size = len(dataset) - train_size
            train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
            trainX, trainY = self.data_set(train)
            testX, testY = self.data_set(test)
            return trainX, trainY, testX, testY


    class CalculateScoreMatrix(Layer):
        def __init__(self, output_dim=None, **kwargs):
            self.output_dim = output_dim
            super(CalculateScoreMatrix, self).__init__(**kwargs)

        def get_config(self):
            config = super().get_config().copy()
            config.update({
                'output_dim': self.output_dim
            })
            return config

        def build(self, input_shape):
            self.kernel = self.add_weight(name='kernel',
                                          shape=(int(input_shape[-1]), int(self.output_dim)),
                                          # initializer='random_normal',
                                          # trainable=True
                                          )
            super(CalculateScoreMatrix, self).build(input_shape)

        def call(self, x):
            res = K.dot(x, self.kernel)
            return res


    class TPALSTM_Model:
        def __init__(self, hidden_unit=100, filter_size=4, batch_size=64, epochs=100, learning_rate=0.0006):
            self.model_path = None
            self.callback_list = None
            self.epochs = epochs
            self.feat_dim = None
            self.input_dim = None
            self.output_dim = None
            self.lr = learning_rate
            self.filters = hidden_unit
            self.batch_size = batch_size
            self.units = hidden_unit
            self.model_name = "cnnlstmtpa"
            self.filter_size = filter_size
            self.learning_rate = learning_rate
            self.loss_fn = losses.mean_squared_error
            self._estimators = {}

        def build_model(self):
            inp = Input(shape=(self.input_dim, self.feat_dim))
            # convolution layer
            x = Conv1D(filters=64, kernel_size=3, strides=1, padding="same")(inp)
            x = Dropout(rate=0.2)(x)
            x = Conv1D(filters=64, kernel_size=7, strides=1, padding="same")(x)
            x = Dropout(rate=0.2)(x)
            # LSTM layer
            x = LSTM(units=self.units, return_sequences=True)(x)
            x = Dropout(rate=0.2)(x)
            x = LSTM(units=self.units, return_sequences=True)(x)
            x = Dropout(rate=0.2)(x)
            # get the 1~t-1 and t hidden state
            H = Lambda(lambda x: x[:, :-1, :])(x)
            x = Dropout(rate=0.2)(x)
            ht = Lambda(lambda x: x[:, -1, :])(x)
            ht = Reshape((self.units, 1))(ht)
            # get the HC by 1*1 convolution
            HC = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1]))(H)
            HC = tf.cast(HC, dtype=tf.float32)
            score_mat = CalculateScoreMatrix(int(self.units))(HC)
            score_mat = Lambda(lambda x: K.batch_dot(x[0], x[1]))([score_mat, ht])
            # get the attn matrix
            score_mat = Activation("sigmoid")(score_mat)
            attn_mat = Multiply()([HC, score_mat])
            attn_vec = Lambda(lambda x: K.sum(x, axis=-1))(attn_mat)
            # get the final prediction
            wvt = Dense(units=self.filters * 4)(attn_vec)
            wht = Dense(units=self.filters * 4)(Flatten()(ht))
            yht = Add()([wht, wvt])
            # get the output
            out = Dense(self.output_dim, activation="linear")(yht)
            # compile
            model = Model(inputs=inp, outputs=out)
            optimizer = optimizers.Adam(lr=self.lr)
            model.compile(loss=self.loss_fn, optimizer=optimizer)
            return model

        def fit(self, x_train, y_train):
            import tensorflow as tf
            # get the dimension of input

            _, input_dim, feat_dim = x_train.shape
            output_dim = 1
            self.input_dim = input_dim
            self.feat_dim = feat_dim
            self.output_dim = output_dim
            # build the model
            my_model = self.build_model()
            # train the model
            sample_weight_train = np.array([_ for _ in range(1, len(x_train) + 1)]) / len(x_train)  # 越靠近当前时间点的数据权重越大
            with TemporaryDirectory() as tmp_dir:
                self.model_path = os.path.join(self.model_name + ".h5")
                self.callback_list = [ModelCheckpoint(filepath="CNN-BILSTM-TPA.h5", monitor='val_loss', verbose=1,
                                                      save_best_only=True, mode="auto", save_weights_only=False)]
                history = my_model.fit(x_train, y_train,
                                       batch_size=self.batch_size,
                                       epochs=self.epochs,
                                       validation_split=0.1,
                                       verbose=1,
                                       sample_weight=sample_weight_train,
                                       callbacks=self.callback_list)
                lossy = history.history['loss']
                np.save("lstm_tpa_loss.npy", np.array(lossy))
                lossy_val = history.history['val_loss']
                np.save("lstm_tpa_valloss.npy", np.array(lossy_val))
                my_model.save(self.model_path)
                # my_model.save('TPA_LSTM_weights.h5')
            # save the every model
            logging.info(f"TPA-LSTM model is saved. ")
            # clear the session and memory
            K.clear_session()
            pass

        def predict(self, x_test):
            trained_model = load_model("CNN-BILSTM-TPA.h5", custom_objects={"CalculateScoreMatrix": CalculateScoreMatrix})
            logging.info(f"The trained model is loaded. ")
            pred_result = trained_model.predict(x_test)
            pred_result = np.reshape(pred_result, (pred_result.size,))
            logging.info(f"The forecast result is got. ")
            return pred_result
model = load_model('cnnlstmtpa.h5',custom_objects={'CalculateScoreMatrix':CalculateScoreMatrix})
#model = load_model('CNN-BILSTM-TPA.h5')
if __name__ == '__main__':
    data1 = [1,2,3,4,5,7,8,9,2]
    dataset1 = np.array(data1).reshape(1, -1, 1)
    testPredict1 = model.predict(dataset1)
    print("over")
    ser.write('over\n'.encode('utf-8'))
    data_list = []
    while True:
        data_read = ser.readline()
        start = time.perf_counter()
        data_str = str(data_read)[2:-3]
        if bool(re.search('[a-zA-Z]',data_str)) == True:
            ser.write('over\n'.encode('utf-8'))
            print("over!!!")
            data_list = []
        else:
            data_float = float(data_str)
            if len(data_list) < 9:
                data_list.append(data_float)
                print(data_list)
                if len(data_list) == 9:
                    dataset = np.array(data_list).reshape(1, -1, 1)  # lookback = 9
                    testPredict = model.predict(dataset)
                    print("list:",data_list)
                    testPredict_str = str(testPredict)[2:-3]
                    print('pre_data', testPredict_str)
                    time_pre = round(time.perf_counter() - start, 3)
                    time_str = str(time_pre) + 'ms'
                    write = str(data_list[8])+' '+testPredict_str + ' ' + time_str + "\n"
                    ser.write(write.encode('utf-8'))
            else:
                for i in range(0,8):
                    data_list[i] = data_list[i+1]
                data_list[8] = data_float
                dataset = np.array(data_list).reshape(1, -1, 1)  # lookback = 9
                testPredict = model.predict(dataset)
                print("list:", data_list)
                testPredict_str = str(testPredict)[2:-3]
                print('pre_data', testPredict_str)
                time_pre = round(time.perf_counter() - start, 3)
                time_str = str(time_pre)+'ms'
                write = str(data_float)+' '+testPredict_str+' '+time_str+"\n"
                ser.write(write.encode('utf-8'))

