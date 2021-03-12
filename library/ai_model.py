import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
import statistics

from tensorflow.keras.callbacks import Callback
from library.logging_pack import *


class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logger.debug(f"End epoch {epoch} of training;")
        keys = list(logs.keys())
        train_logs = []
        for key in keys:
            train_logs.append(logs[f"{key}"])
        # logs에 None 값이 포함되어 있으면 학습 종료
        if None in train_logs:
            logger.debug(f"{train_logs}에 None 값이 존재하여 학습 종료")
            self.model.stop_training = True


def back_testing(code, test_sample_df, y_pred, ai_settings, history, acc, date_start):
    # 3단계
    lstm_book_df = test_sample_df[['close', 'next_rtn']].copy()
    t1 = DataFrame(data=y_pred, columns=['position'], index=lstm_book_df.index[5:])
    lstm_book_df = lstm_book_df.join(t1, how='left')
    lstm_book_df.fillna(0, inplace=True)
    lstm_book_df['ret'] = lstm_book_df['close'].pct_change()
    lstm_book_df['lstm_ret'] = lstm_book_df['next_rtn'] * lstm_book_df['position'].shift(1)
    lstm_book_df['lstm_cumret'] = (lstm_book_df['lstm_ret'] + 1).cumprod()
    lstm_book_df['bm_cumret'] = (lstm_book_df['ret'] + 1).cumprod()

    lstm_book_df[['lstm_cumret', 'bm_cumret']].plot()

    # Backtesting
    historical_max = lstm_book_df['close'].cummax()
    daily_drawdown = lstm_book_df['close'] / historical_max - 1.0
    historical_dd = daily_drawdown.cummin()
    historical_dd.plot()

    # BM 바이앤홀드
    CAGR = lstm_book_df.loc[lstm_book_df.index[-1], 'bm_cumret'] ** (252. / len(lstm_book_df.index)) - 1
    Sharpe = np.mean(lstm_book_df['ret']) / np.std(lstm_book_df['ret']) * np.sqrt(252.)
    VOL = np.std(lstm_book_df['ret']) * np.sqrt(252.)
    MDD = historical_dd.min()
    bm_text = f"""[BM Buy and Hold]\n
CAGR : {round(CAGR * 100, 2)}%
Sharpe : {round(Sharpe, 2)}
VOL : {round(VOL * 100, 2)}%
MDD : {round(-1 * MDD * 100, 2)}%"""

    # LSTM
    CAGR = lstm_book_df.loc[lstm_book_df.index[-1], 'lstm_cumret'] ** (252. / len(lstm_book_df.index)) - 1
    Sharpe = np.mean(lstm_book_df['lstm_ret']) / np.std(lstm_book_df['lstm_ret']) * np.sqrt(252.)
    VOL = np.std(lstm_book_df['lstm_ret']) * np.sqrt(252.)
    MDD = historical_dd.min()
    lstm_text = f"""[LSTM]\n
CAGR : {round(CAGR * 100, 2)}%
Sharpe : {round(Sharpe, 2)}
VOL : {round(VOL * 100, 2)}%
MDD : {round(-1 * MDD * 100, 2)}%

========================================
mean loss: 
{statistics.mean(history.history['loss'])}
{history.history['loss']}

mean acc: 
{statistics.mean(history.history['accuracy'])}
{history.history['accuracy']}

mean val_loss: 
{statistics.mean(history.history['val_loss'])}
{history.history['val_loss']}

mean val_acc: 
{statistics.mean(history.history['val_accuracy'])}
last val_acc: 
{history.history['val_accuracy'][-1]}
MAX: 
{max(history.history['val_accuracy'])}
MIN: 
{min(history.history['val_accuracy'])}
{history.history['val_accuracy']}\n"""

    fcode = code.replace('/', '-')
    # f = open(f"history/{fcode}.txt", 'w')
    folder_name = f'history/{date_start}'
    fname = f"""{folder_name}/{fcode}_{ai_settings['table']}_epoch_{ai_settings['epochs']}_nstep_{ai_settings['num_step']}\
_units_{ai_settings['num_units']}_batch_{ai_settings['batch_size']}\
_learning_rate_{str(ai_settings['learning_rate']).replace('0.', '')}_optimizer_{ai_settings['optimizer']}\
_loss_{ai_settings['loss']}_activation_{ai_settings['activation']}.txt"""
    data = f"{bm_text}\n\n{lstm_text}\n\n모델 정확도:\n{float(acc)*100}%\n{ai_settings}"
    logger.debug(lstm_text)
    encoded_data = data.encode("utf8")
    f = open(f"{fname}", 'wb')
    f.write(encoded_data)
    f.close()


def plot_train_test(train_sample_df, test_sample_df):
    plt.figure(figsize=(16, 5))
    ax = plt.subplot(1, 2, 1)
    plt.plot(train_sample_df['close'])
    plt.title("Train")
    ax = plt.subplot(1, 2, 2)
    plt.plot(test_sample_df['close'])
    plt.title("Test")


def plot_history(history):
    plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.title("Train loss")
    ax = plt.subplot(1, 2, 2)
    plt.plot(history.history["val_loss"])
    plt.title("Test loss")
    plt.savefig('sample.png')


def create_model(x_train, ai_settings):
    K.clear_session()
    # 입력데이터셋 형태에 맞게 지정
    # 케라스에서 첫번째 차원에는 데이터의 개수가 들어가는데, 임의의 스칼라를 의미하는 None 값을 넣어준다
    # 두번째 차원에는 데이터의 시간축(time_step), 세번째는 LSTM 입력층에 입력되는 특성 데이터 (feature) 개수
    input_layer = Input(batch_shape=(None, x_train.shape[1], x_train.shape[2]))
    # 다층 구조로 LSTM 층 위에 LSTM 층이 연결: LSTM(input)(input_layer)
    # return_sequences=True 이전 layer의 출력이 다음 layer의 입력으로 전달
    layer_lstm_1 = LSTM(ai_settings['num_units'], return_sequences=True, recurrent_regularizer=regularizers.l2(ai_settings['learning_rate']))(input_layer)
    layer_lstm_1 = BatchNormalization()(layer_lstm_1)  # 배치정규화층을 이어준다
    layer_lstm_2 = LSTM(ai_settings['num_units'], return_sequences=True, recurrent_regularizer=regularizers.l2(ai_settings['learning_rate']))(layer_lstm_1)
    layer_lstm_2 = Dropout(0.25)(layer_lstm_2)  # 드롭아웃하여 임의의 확률로 가중치 선을 삭제 - 과적합 방지
    layer_lstm_3 = LSTM(ai_settings['num_units'], return_sequences=True, recurrent_regularizer=regularizers.l2(ai_settings['learning_rate']))(layer_lstm_2)
    layer_lstm_3 = BatchNormalization()(layer_lstm_3)
    layer_lstm_4 = LSTM(ai_settings['num_units'], return_sequences=True, recurrent_regularizer=regularizers.l2(ai_settings['learning_rate']))(layer_lstm_3)
    layer_lstm_4 = Dropout(0.25)(layer_lstm_4)
    layer_lstm_5 = LSTM(ai_settings['num_units'], recurrent_regularizer=regularizers.l2(ai_settings['learning_rate']))(layer_lstm_4)
    layer_lstm_5 = BatchNormalization()(layer_lstm_5)
    output_layer = Dense(2, activation=ai_settings['activation'])(layer_lstm_5)
    '''
    Dense: 완전 연결층으로 연결되면서 최종 예측값을 출력
    Dense 첫번째 인자 = units = 출력 뉴런의 수.
    activation = 활성화 함수
    '''
    # 입력층과 출력층을 연결해 모델 객체를 만들어낸다.
    model = Model(input_layer, output_layer)
    # 모델 학습 방식을 설정하여 모델 객체를 만들어낸다. (손실함수, 최적화함수, 모델 성능 판정에 사용되는 지표)
    model.compile(loss=ai_settings['loss'], optimizer=ai_settings['optimizer'], metrics=['accuracy'])
    print(model.summary())
    return model


def create_dataset_binary(data, feature_list, ai_settings, n_feature):
    '''
    다음날 시종가 수익률 라벨링.
    '''
    # LSTM 모델에 넣을 변수 데이터 선택
    train_xdata = np.array(data[feature_list[0:n_feature]])
    # 레이블링 데이터를 만든다, next_rtn 값
    train_ydata = np.array(data[[feature_list[n_feature]]])
    step = ai_settings['num_step']
    # 마지막 단계
    m = np.arange(len(train_xdata) - step)
    x, y = [], []
    for i in m:
        # 각 단계마다 사용할 학습 데이터 기간 정의 (step = 얼마만큼의 데이터 기간을 입력값으로 전달할지)
        a = train_xdata[i:(i + step)]
        x.append(a)
    # 신경망 학습을 할 수 있도록 3차원 데이터 형태로 구성: batch_size: len(m), 시퀀스 내 행의 개수: step, feature 개수 (열의 개수): n
    # data:np.array(x), (len(m), 5, 8) = (len(x_batch), len(x_batch[0]), len(x_batch[0][0]))
    x_batch = np.reshape(np.array(x), (len(m), step, n_feature))

    # n일 뒤 데이터 예측
    n_pred = ai_settings['n_pred'] - 1
    if n_pred == 0:
        label_len = m + step
    else:
        label_len = m[:-n_pred] + step

    for i in label_len:
        # 이진 분류르 하기 위한 next_rtn
        next_rtn = train_ydata[i + n_pred][0]
        # 이진 분류: next_rtn가 0보다 크면 다음날 오를 것이라고 가정하여 해당 방향성을 레이블로 설정
        if next_rtn > 0:
            label = 1
        else:
            label = 0
        # 순차적으로 임시로 생성된 레이블을 저장
        y.append(label)
    # 학습을 위한 1차원 열 벡터 형대로 변환 : (662,) -> (662, 1)
    y_batch = np.reshape(np.array(y), (-1, 1))
    y_batch = to_categorical(y_batch, 2)

    # x_batch와 y_batch 길이 맞추기
    if n_pred == 0:
        return x_batch, y_batch
    else:
        return x_batch[:-n_pred], y_batch


def min_max_normal(tmp_df, all_features, feature4_list):
    eng_list = []
    sample_df = tmp_df.copy()
    for x in all_features:
        if x in feature4_list:
            continue
        series = sample_df[x].copy()
        values = series.values
        values = values.reshape((len(values), 1))
        # train the normalization. sklearn 라이브러리에서 정규화 객체를 받는다.
        scaler = MinMaxScaler(feature_range=(0, 1))
        # 입력 데이터에 대해 정규화 범위를 탐색
        scaler = scaler.fit(values)
        # 데이터셋 정규화 및 출력 (최소-최대 정규화)
        normalized = scaler.transform(values)
        # 정규화된 데이터를 새로운 컬럼명으로 저장
        new_feature = f'{x}_normal'
        eng_list.append(new_feature)
        sample_df[new_feature] = normalized
    return sample_df, eng_list


def data_split(df):
    '''
     np.split will split at 60% of the length of the shuffled array,
     then 80% of length (which is an additional 20% of data),
     thus leaving a remaining 20% of the data. This is due to the definition of the function.
     You can test/play with: x = np.arange(10.0), followed by np.split(x, [ int(len(x)*0.6), int(len(x)*0.8)])
    '''
    # produces a 60%, 20%, 20% split for training, validation and test sets.
    train_df, val_df, test_df = np.split(df, [int(.6 * len(df)), int(.8 * len(df))]).copy()
    return train_df, val_df, test_df
