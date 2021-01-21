import pandas as pd
import pandas_datareader as pdr
import talib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import random as python_random

seed_number = 7
np.random.seed(seed_number)
python_random.seed(seed_number)
# tf.set_random_seed(seed_number)  # v1
tf.random.set_seed(seed_number)  # v2

# data load
df = pd.read_csv('./data/intc.csv',
                 index_col='Date',
                 parse_dates=True)
sox_df = pd.read_csv('./data/sox_df.csv',
                     index_col='Date',
                     parse_dates=True)
vix_df = pd.read_csv('./data/vix_df.csv',
                     index_col='Date',
                     parse_dates=True)
snp500_df = pd.read_csv('./data/s&p500.csv',
                        index_col='Date',
                        parse_dates=True)

# data features
df['next_price'] = df['Adj Close'].shift(-1)
df['next_rtn'] = df['Close'] / df['Open'] - 1
df['log_return'] = np.log(1 + df['Adj Close'].pct_change())
df['CCI'] = talib.CCI(df['High'], df['Low'], df['Adj Close'], timeperiod=14)

# 1.RA : Standard deviation rolling average
# Moving Average
df['MA5'] = talib.SMA(df['Close'], timeperiod=5)
df['MA10'] = talib.SMA(df['Close'], timeperiod=10)
df['RASD5'] = talib.SMA(talib.STDDEV(df['Close'], timeperiod=5, nbdev=1), timeperiod=5)
df['RASD10'] = talib.SMA(talib.STDDEV(df['Close'], timeperiod=5, nbdev=1), timeperiod=10)

# 2.MACD : Moving Average Convergence/Divergence
macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['MACD'] = macd

# Momentum Indicators
# 3.CCI : Commodity Channel Index
df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
# Volatility Indicators

# 4.ATR : Average True Range
df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

# 5.BOLL : Bollinger Band
upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
df['ub'] = upper
df['middle'] = middle
df['lb'] = lower

# 7.MTM1
df['MTM1'] = talib.MOM(df['Close'], timeperiod=1)

# 7.MTM3
df['MTM3'] = talib.MOM(df['Close'], timeperiod=3)

# 8.ROC : Rate of change : ((price/prevPrice)-1)*100
df['ROC'] = talib.ROC(df['Close'], timeperiod=60)

# 9.WPR : william percent range (Williams' %R)
df['WPR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)

snp500_df = snp500_df.loc[:, ['Close']].copy()
snp500_df.rename(columns={'Close': 'S&P500'}, inplace=True)
sox_df = sox_df.loc[:, ['Close']].copy()
sox_df.rename(columns={'Close': 'SOX'}, inplace=True)
vix_df = vix_df.loc[:, ['Close']].copy()
vix_df.rename(columns={'Close': 'VIX'}, inplace=True)

df = df.join(snp500_df, how='left')
df = df.join(sox_df, how='left')
df = df.join(vix_df, how='left')

df.head()

# feature list
# feature_list = ['Adj Close', 'log_return', 'CCI','next_price']
# 볼린저 밴드와 MACD를 어떻게 활용해야할까? 음. 아님 그냥 그대로 사용하는 건가?
feature1_list = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'log_return']
feature2_list = ['RASD5', 'RASD10', 'ub', 'lb', 'CCI', 'ATR', 'MACD', 'MA5', 'MA10', 'MTM1', 'MTM3', 'ROC', 'WPR']
feature3_list = ['S&P500', 'SOX', 'VIX']
# feature4_list = ['next_price']
feature4_list = ['next_rtn']

all_features = feature1_list + feature2_list + feature3_list + feature4_list

phase_flag = '3'

if phase_flag == '1':
    train_from = '2010-01-04'
    train_to = '2012-01-01'

    val_from = '2012-01-01'
    val_to = '2012-04-01'

    test_from = '2012-04-01'
    test_to = '2012-07-01'

elif phase_flag == '2':
    train_from = '2012-07-01'
    train_to = '2014-07-01'

    val_from = '2014-07-01'
    val_to = '2014-10-01'

    test_from = '2014-10-01'
    test_to = '2015-01-01'

else:
    train_from = '2015-01-01'
    train_to = '2017-01-01'

    val_from = '2017-01-01'
    val_to = '2017-04-01'

    test_from = '2017-04-01'
    test_to = '2017-07-01'

# train / validation / testing
train_df = df.loc[train_from:train_to, all_features].copy()
val_df = df.loc[val_from:val_to, all_features].copy()
test_df = df.loc[test_from:test_to, all_features].copy()


def min_max_normal(tmp_df):
    eng_list = []
    sample_df = tmp_df.copy()
    for x in all_features:
        if x in feature4_list:
            continue
        series = sample_df[x].copy()
        values = series.values
        values = values.reshape((len(values), 1))
        # train the normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(values)
        #         print('columns : %s , Min: %f, Max: %f' % (x, scaler.data_min_, scaler.data_max_))
        # normalize the dataset and print
        normalized = scaler.transform(values)
        new_feature = '{}_normal'.format(x)
        eng_list.append(new_feature)
        sample_df[new_feature] = normalized
    return sample_df, eng_list


train_sample_df, eng_list = min_max_normal(train_df)
val_sample_df, eng_list = min_max_normal(val_df)
test_sample_df, eng_list = min_max_normal(test_df)
# train_sample_df.head()

# lstm model 훈련 데이터 구분
num_step = 5
num_unit = 200


def create_dateset_binary(data, feature_list, step, n):
    '''
    다음날 시종가 수익률 라벨링.
    '''
    train_xdata = np.array(data[feature_list[0:n]])

    # 가장 뒤 n step을 제외하기 위해. 왜냐하면 학습 input으로는 어차피 10개만 주려고 하니깐.
    m = np.arange(len(train_xdata) - step)
    #     np.random.shuffle(m)  # shufflee은 빼자.
    x, y = [], []
    for i in m:
        a = train_xdata[i:(i + step)]
        x.append(a)
    x_batch = np.reshape(np.array(x), (len(m), step, n))

    train_ydata = np.array(data[[feature_list[n]]])
    # n_step 이상부터 답을 사용할 수 있는거니깐.
    for i in m + step:
        next_rtn = train_ydata[i][0]
        if next_rtn > 0:
            label = 1
        else:
            label = 0
        y.append(label)
    y_batch = np.reshape(np.array(y), (-1, 1))
    return x_batch, y_batch


eng_list = eng_list + feature4_list
n_feature = len(eng_list) - 1
# LSTM할때 사용했던 소스코드.
x_train, y_train = create_dateset_binary(train_sample_df[eng_list], eng_list, num_step, n_feature)
x_val, y_val = create_dateset_binary(val_sample_df[eng_list], eng_list, num_step, n_feature)
x_test, y_test = create_dateset_binary(test_sample_df[eng_list], eng_list, num_step, n_feature)

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, 2)
y_val = to_categorical(y_val, 2)
y_test = to_categorical(y_test, 2)

# 모델 생성
x_train.shape[1]
# 이것은 전체 데이터를 242 rolling , 10 window, 2개 feature를 본다는 것이다.
# 2개 feature를 10개 묶음으로 보는데, 1칸씩 미루면서 보니 242개 데이터를 본다는 것이다.

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

K.clear_session()
input_layer = Input(batch_shape=(None, x_train.shape[1], x_train.shape[2]))
layer_lstm_1 = LSTM(num_unit, return_sequences=True, recurrent_regularizer=regularizers.l2(0.01))(input_layer)
layer_lstm_1 = BatchNormalization()(layer_lstm_1)
layer_lstm_2 = LSTM(num_unit, return_sequences=True, recurrent_regularizer=regularizers.l2(0.01))(layer_lstm_1)
layer_lstm_2 = Dropout(0.25)(layer_lstm_2)
layer_lstm_3 = LSTM(num_unit, return_sequences=True, recurrent_regularizer=regularizers.l2(0.01))(layer_lstm_2)
layer_lstm_3 = BatchNormalization()(layer_lstm_3)
layer_lstm_4 = LSTM(num_unit, return_sequences=True, recurrent_regularizer=regularizers.l2(0.01))(layer_lstm_3)
layer_lstm_4 = Dropout(0.25)(layer_lstm_4)
layer_lstm_5 = LSTM(num_unit, recurrent_regularizer=regularizers.l2(0.01))(layer_lstm_4)
layer_lstm_5 = BatchNormalization()(layer_lstm_5)
output_layer = Dense(2, activation='sigmoid')(layer_lstm_5)

model = Model(input_layer, output_layer)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# 모델 학습
history = model.fit(x_train, y_train, epochs=20, batch_size=10, validation_data=(x_val, y_val))


# loss 그래프
def plot_history(history):
    plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.title("Train loss")
    ax = plt.subplot(1, 2, 2)
    plt.plot(history.history["val_loss"])
    plt.title("Test loss")
    plt.savefig('sample.png')


plot_history(history)  # 3단계
# ? 이렇게 모델을 저장하는 건가?
# model.save('model_functional_open_close_binary_phase3.h5')

# 예측
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

predicted = model.predict(x_test)
y_pred = np.argmax(predicted, axis=1)
Y_test = np.argmax(y_test, axis=1)
print(len(y_test))
print(len(y_pred))

plt.figure(figsize=(16, 5))
ax = plt.subplot(1, 2, 1)
plt.plot(train_sample_df['Adj Close'])
plt.title("Train")
ax = plt.subplot(1, 2, 2)
plt.plot(test_sample_df['Adj Close'])
plt.title("Test")

# 3단계
lstm_book_df = test_sample_df[['Adj Close', 'next_rtn']].copy()
t1 = pd.DataFrame(data=y_pred, columns=['position'], index=lstm_book_df.index[5:])
lstm_book_df = lstm_book_df.join(t1, how='left')
lstm_book_df.fillna(0, inplace=True)
lstm_book_df['ret'] = lstm_book_df['Adj Close'].pct_change()
lstm_book_df['lstm_ret'] = lstm_book_df['next_rtn'] * lstm_book_df['position'].shift(1)
lstm_book_df['lstm_cumret'] = (lstm_book_df['lstm_ret'] + 1).cumprod()
lstm_book_df['bm_cumret'] = (lstm_book_df['ret'] + 1).cumprod()

lstm_book_df[['lstm_cumret', 'bm_cumret']].plot()

# Backtesting
historical_max = lstm_book_df['Adj Close'].cummax()
daily_drawdown = lstm_book_df['Adj Close'] / historical_max - 1.0
historical_dd = daily_drawdown.cummin()
historical_dd.plot()

# BM 바이앤홀드
CAGR = lstm_book_df.loc[lstm_book_df.index[-1], 'bm_cumret'] ** (252. / len(lstm_book_df.index)) - 1
Sharpe = np.mean(lstm_book_df['ret']) / np.std(lstm_book_df['ret']) * np.sqrt(252.)
VOL = np.std(lstm_book_df['ret']) * np.sqrt(252.)
MDD = historical_dd.min()
print('CAGR : ', round(CAGR * 100, 2), '%')
print('Sharpe : ', round(Sharpe, 2))
print('VOL : ', round(VOL * 100, 2), '%')
print('MDD : ', round(-1 * MDD * 100, 2), '%')
# LSTM
CAGR = lstm_book_df.loc[lstm_book_df.index[-1], 'lstm_cumret'] ** (252. / len(lstm_book_df.index)) - 1
Sharpe = np.mean(lstm_book_df['lstm_ret']) / np.std(lstm_book_df['lstm_ret']) * np.sqrt(252.)
VOL = np.std(lstm_book_df['lstm_ret']) * np.sqrt(252.)
MDD = historical_dd.min()
print('CAGR : ', round(CAGR * 100, 2), '%')
print('Sharpe : ', round(Sharpe, 2))
print('VOL : ', round(VOL * 100, 2), '%')
print('MDD : ', round(-1 * MDD * 100, 2), '%')
