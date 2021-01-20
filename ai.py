import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Activation, BatchNormalization, Dropout
from tensorflow.keras import backend as K, regularizers


def create_model(x_train, num_unit):
    # LSTM 모델을 생성한다.
    K.clear_session()
    # 입력데이터셋 형태에 맞게 지정
    # 케라스에서 첫번째 차원에는 데이터의 개수가 들어가는데, 임의의 스칼라를 의미하는 None 값을 넣어준다
    # 두번째 차원에는 데이터의 시간축(time_step), 세번째는 LSTM 입력층에 입력되는 특성 데이터 (feature) 개수
    input_layer = Input(batch_shape=(None, x_train.shape[1], x_train.shape[2]))
    # 다층 구조로 LSTM 층 위에 LSTM 층이 연결: LSTM(input)(input_layer)
    # return_sequences=True 이전 layer의 출력이 다음 layer의 입력으로 전달
    layer_lstm1 = LSTM(num_unit, return_sequences=True, recurrent_regularizer=regularizers.l2(0.01))(input_layer)
    # 배치정규화층을 이어준다
    layer_lstm1 = BatchNormalization()(layer_lstm1)

    layer_lstm2 = LSTM(num_unit, return_sequences=True, recurrent_regularizer=regularizers.l2(0.01))(layer_lstm1)
    # 드롭아웃하여 임의의 확률로 가중치 선을 삭제 - 과적합 방지
    layer_lstm2 = Dropout(0.25)(layer_lstm2)

    layer_lstm3 = LSTM(num_unit, return_sequences=True, recurrent_regularizer=regularizers.l2(0.01))(layer_lstm2)
    layer_lstm3 = BatchNormalization()(layer_lstm3)

    layer_lstm4 = LSTM(num_unit, return_sequences=True, recurrent_regularizer=regularizers.l2(0.01))(layer_lstm3)
    layer_lstm4 = Dropout(0.25)(layer_lstm4)

    layer_lstm5 = LSTM(num_unit, return_sequences=True, recurrent_regularizer=regularizers.l2(0.01))(layer_lstm4)
    layer_lstm5 = BatchNormalization()(layer_lstm5)
    # Dense: 완전 연결층으로 연결되면서 최종 예측값을 출력
    '''
    Dense 첫번째 인자 = units = 출력 뉴런의 수.
    input_dim = 입력 뉴런의 수. (입력의 차원)
    activation = 활성화 함수.
    - linear  : 디폴트 값으로 별도 활성화 함수 없이 입력 뉴런과 가중치의 계산 결과 그대로 출력. Ex) 선형 회귀
    - sigmoid : 이진 분류 문제에서 출력층에 주로 사용되는 활성화 함수. 레이어가 깊어질 수록 그라이언트가 전달되지 않는 vanishing gradient 문제가 발생 (학습이 안되는 상황)
    - softmax : 셋 이상을 분류하는 다중 클래스 분류 문제에서 출력층에 주로 사용되는 활성화 함수.
    - relu : 은닉층에 주로 사용되는 활성화 함수.
    '''
    output_layer = Dense(1, activation='sigmoid')(layer_lstm5)
    # output_layer = Dense(1, activation='softmax')(layer_lstm5) # 단점보완 LeakyReLU (일반적으로 알파를 0.01로 설정)
    # RNN, LSTM 등을 학습시킬 때 사용
    # output_layer = Dense(1, activation='tanh')(layer_lstm5)
    # CNN을 학습시킬 때 많이 사용, 0 이하의 값은 다음 레이어에 전달하지 않는다. 계산식 매우 간단함으로써 연산 속도가 빨라질 수 있고, 구현하기 편하다
    # output_layer = Dense(1, activation='relu')(layer_lstm5) # 단점보완 LeakyReLU (일반적으로 알파를 0.01로 설정)

    # 입력층과 출력층을 연결해 모델 객체를 만들어낸다.
    model = Model(input_layer, output_layer)
    # 모델 학습 방식을 설정하여 모델 객체를 만들어낸다. (손실함수, 최적화함수, 모델 성능 판정에 사용되는 지표)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # binary_crossentropy - sigmoid, categorical_crossentropy - softmax 이 조합으로 사용된다.
    # 레이블 클래스가 두 개 뿐인 경우 (0과 1로 가정)이 교차 엔트로피 손실을 사용
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Debug model.summary()
    return model


def create_dataset_binary(data, feature_list, step, n):
    # LSTM 모델에 넣을 변수 데이터 선택
    train_xdata = np.array(data[feature_list[:n]])
    # 마지막 단계
    m = np.arange(len(train_xdata) - step)
    x, y = [], []
    for i in m:
        # 각 단계마다 사용할 학습 데이터 기간 정의 (step = 얼마만큼의 데이터 기간을 입력값으로 전달할지)
        a = train_xdata[i:(i + step)]
        x.append(a)
    # 신경망 학습을 할 수 있도록 3차원 데이터 형태로 구성: batch_size: len(m), 시퀀스 내 행의 개수: step, feature 개수 (열의 개수): n
    x_batch = np.reshape(np.array(x), (len(m), step, n))

    # 레이블링 데이터를 만든다. (레이블 데이터는 다음날 종가)
    train_ydata = np.array(data[feature_list[n - 5]])  # Close_normal 값
    # n_step 이상부터 답을 사용
    for i in m + step:
        # 이진 분류를 하기 위한 시작 종가
        start_price = train_ydata[i - 1]
        # 이진 분류르 하기 위한 종료 종가
        end_price = train_ydata[i]

        # 이진 분류: 종료 종가가 더 크면 다음날 오를 것이라고 가정하여 해당 방향성을 레이블로 설정
        if end_price > start_price:
            label = 1  # 오르면 1
        else:
            label = 0  # 떨어지면 0
        # 임시로 생성된 레이블을 순차적으로 저장
        y.append(label)
    # 학습을 위한 1차원 열 벡터 형대로 변환 : (662,) -> (662, 1)
    y_batch = np.reshape(np.array(y), (-1, 1))
    # x_batch.shape: (662, 5, 7), y_batch.shape: (662, 1)
    return x_batch, y_batch


def min_max_normal(tmp_df):
    eng_list = []
    sample_df = tmp_df.copy()
    all_features = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ubb', 'mbb', 'lbb', 'next_rtn']
    for x in all_features:
        if x in ('Date', 'next_rtn'):
            continue
        series = sample_df[x].copy()
        values = series.values
        values = values.reshape((len(values), 1))
        # 스케일러생성 및 훈련
        # sklearn 라이브러리에서 정규화 객체를 받는다.
        scaler = MinMaxScaler(feature_range=(0, 1))
        # 입력 데이터에 대해 정규화 범위를 탐색
        scaler = scaler.fit(values)
        # 데이터셋 정규화 및 출력
        # 입력데이터를 최소-최대 정규화
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


def _data_split(examples, labels, train_frac=0.6, random_state=None):
    ''' https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    param data:       Data to be split
    param train_frac: Ratio of train set to whole dataset

    Randomly split dataset, based on these ratios:
        'train': train_frac
        'valid': (1-train_frac) / 2
        'test':  (1-train_frac) / 2

    Eg: passing train_frac=0.8 gives a 80% / 10% / 10% split
    '''

    assert 0 < train_frac <= 1, "Invalid training set fraction"

    X_train, X_tmp, Y_train, Y_tmp = train_test_split(
        examples, labels, train_size=train_frac, random_state=random_state)

    X_val, X_test, Y_val, Y_test = train_test_split(
        X_tmp, Y_tmp, train_size=0.5, random_state=random_state)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test
