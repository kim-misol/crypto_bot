import numpy as np

from ai_model import data_split, min_max_normal, create_dataset_binary, create_model, back_testing
from graphs import plot_model_fit_history


def train_model(coin_df, code):
    coin_df['next_rtn'] = coin_df['Close'] / coin_df['Open'] - 1
    # 학습, 검증, 테스트 데이터 기간 분할 6:2:2
    train_df, val_df, test_df = data_split(coin_df)
    default_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    # add_features = ['ubb', 'mbb', 'lbb']
    add_features = []
    extra_list = ['next_rtn']
    all_features = default_features + add_features + extra_list

    # 최소-최대 정규화
    train_sample_df, eng_list = min_max_normal(train_df, all_features, extra_list)
    # n일 이동평균값 이용 시 nan 값 있는 데이터 제거, nan 값 제거 후 model.fit 할 때 loss도 정상 출력, 에측값도 정상
    nan_cnt = np.where(np.isnan(train_sample_df))[0][-1] + 1
    train_sample_df = train_sample_df[nan_cnt:]
    val_sample_df, eng_list = min_max_normal(val_df, all_features, extra_list)
    test_sample_df, eng_list = min_max_normal(test_df, all_features, extra_list)

    # 레이블링 테이터
    # (num_step)일치 (n_feature)개 변수의 데이터를 사용해 다음날 data 예측
    num_step = 5
    num_unit = 200
    eng_list = eng_list + extra_list
    n_feature = len(eng_list) - 1  # next_rtn 을 제외한 feature 개수

    # 훈련, 검증, 테스트 데이터를 변수 데이터와 레이블 데이터로 나눈다
    x_train, y_train = create_dataset_binary(train_sample_df, eng_list, num_step, n_feature)
    x_val, y_val = create_dataset_binary(val_sample_df, eng_list, num_step, n_feature)
    x_test, y_test = create_dataset_binary(test_sample_df, eng_list, num_step, n_feature)

    # model 생성
    model = create_model(x_train, num_unit)
    # ? 모델 저장하여 재사용 가능한지 확인
    # model 학습. 휸련데이터샛을 이용해 epochs만큼 반복 훈련 (논문에선 5000으로 설정). verbose 로그 출력 설정
    # validation_data를 총해 에폭이 끝날 때마다 학습 모델을 해당 데이터로 평가한다. 해당 데이터로 학습하지는 않는다.
    EPOCHS = 20
    BATCH_SIZE = 10
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val))
    fig = plot_model_fit_history(history, EPOCHS)
    fig.show()

    # model 저장
    # model.save(f'{code}_model_functional_open_close_binary_{EPOCHS}.h5')
    model.save(f'{code}_model_functional_open_close_binary_epoch{EPOCHS}.h5')
    # 내일 오를지 내릴지에 대한 label 예측 값 출력
    predicted = model.predict(x_test)
    y_pred = np.argmax(predicted, axis=1)
    Y_test = np.argmax(y_test, axis=1)

    back_testing(code, test_sample_df, y_pred)
    # 모델 리턴해서 내일 오른다는 예측이 있을 경우 매수 떨어지면 매도
    return y_pred


def use_model(coin_df, code):
    coin_df['next_rtn'] = coin_df['Close'] / coin_df['Open'] - 1
    # all_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'ubb', 'mbb', 'lbb', 'next_rtn']
    default_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    # add_features = ['ubb', 'mbb', 'lbb']
    add_features = ['ubb', 'mbb', 'lbb']
    extra_list = ['next_rtn']
    all_features = default_features + add_features + extra_list

    # 최소-최대 정규화
    coin_sample_df, eng_list = min_max_normal(coin_df, all_features, extra_list)
    # n일 이동평균값 이용 시 nan 값 있는 데이터 제거, nan 값 제거 후 model.fit 할 때 loss도 정상 출력, 에측값도 정상
    nan_cnt = np.where(np.isnan(coin_sample_df))[0][-1] + 1
    train_sample_df = coin_sample_df[nan_cnt:]

    # 레이블링 테이터
    # (num_step)일치 (n_feature)개 변수의 데이터를 사용해 다음날 종가 예측
    num_step = 5
    eng_list = eng_list + extra_list
    n_feature = len(eng_list) - 1  # next_rtn 을 제외한 feature 개수

    # 훈련, 검증, 테스트 데이터를 변수 데이터와 레이블 데이터로 나눈다
    x_test, y_test = create_dataset_binary(train_sample_df, eng_list, num_step, n_feature)

    # 모델 불러오기
    from tensorflow.keras.models import load_model
    model = load_model(f'{code}_model_functional_open_close_binary_epoch200.h5')

    # 예측
    predicted = model.predict(x_test)
    y_pred = np.argmax(predicted, axis=1)

    return y_pred