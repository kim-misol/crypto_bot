import numpy as np

from library.ai_model import data_split, min_max_normal, create_dataset_binary, create_model, back_testing
from library.graphs import plot_model_fit_history
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def train_model(ai_filter_num, df, code):
    if ai_filter_num == 1001:
        ai_settings = {
            "table": "min1",
            "num_step": 5,
            "num_units": 200,
            "epochs": 400,
            "batch_size": 10,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "sigmoid",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1002:
        ai_settings = {
            "table": "min1",
            "num_step": 5,
            "num_units": 200,
            "epochs": 400,
            "batch_size": 10,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "softmax",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1003:
        ai_settings = {
            "table": "min1",
            "num_step": 5,
            "num_units": 200,
            "epochs": 200,
            "batch_size": 10,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "tanh",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1004:
        ai_settings = {
            "table": "min1",
            "num_step": 5,
            "num_units": 200,
            "epochs": 400,
            "batch_size": 10,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "tanh",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1:
        ai_settings = {
            "table": "min1",
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 10,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "sigmoid",
            "is_continuously_train": False
        }
    elif ai_filter_num == 2:
        ai_settings = {
            "table": "min1",
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 10,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "relu",
            "is_continuously_train": False
        }
    elif ai_filter_num == 999:
        ai_settings = {
            "table": "min1",
            "num_step": 5,
            "num_units": 200,
            "epochs": 10,
            "batch_size": 10,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "sigmoid",
            "is_continuously_train": False
        }
    else:
        print("ai_filter_num 설정 오류")
        exit(1)

    coin_df = df.copy()
    # 데이터가 10000개(10000일 or 10000분)가 넘지 않으면 예측도가 떨어지기 때문에 학습하지 않는다
    if len(coin_df) < 10000:
        print(f"테스트 데이터가 적어 학습 제외")
        exit(1)

    coin_df['next_rtn'] = coin_df['close'] / coin_df['open'] - 1
    # 학습, 검증, 테스트 데이터 기간 분할 6:2:2
    train_df, val_df, test_df = data_split(coin_df)
    default_features = ['open', 'high', 'low', 'close', 'volume']
    # add_features = ['ubb', 'mbb', 'lbb']
    add_features = []
    extra_list = ['next_rtn']
    all_features = default_features + add_features + extra_list

    # 최소-최대 정규화
    train_sample_df, eng_list = min_max_normal(train_df, all_features, extra_list)
    # n일 이동평균값 이용 시 nan 값 있는 데이터 제거, nan 값 제거 후 model.fit 할 때 loss도 정상 출력, 에측값도 정상
    # nan_cnt = np.where(np.isnan(train_sample_df))[0][-1] + 1
    if len(np.where(np.isnan(train_sample_df))[0]) == 0:
        pass
    else:
        nan_cnt = np.where(np.isnan(train_sample_df))[0][-1] + 1
        train_sample_df = train_sample_df[nan_cnt:]
    val_sample_df, eng_list = min_max_normal(val_df, all_features, extra_list)
    test_sample_df, eng_list = min_max_normal(test_df, all_features, extra_list)

    # 레이블링 테이터
    # (num_step)일치 (n_feature)개 변수의 데이터를 사용해 다음날 data 예측
    eng_list = eng_list + extra_list
    n_feature = len(eng_list) - 1  # next_rtn 을 제외한 feature 개수

    # 훈련, 검증, 테스트 데이터를 변수 데이터와 레이블 데이터로 나눈다
    x_train, y_train = create_dataset_binary(train_sample_df, eng_list, ai_settings, n_feature)
    x_val, y_val = create_dataset_binary(val_sample_df, eng_list, ai_settings, n_feature)
    x_test, y_test = create_dataset_binary(test_sample_df, eng_list, ai_settings, n_feature)

    # 50번이상 더 좋은 결과가 없으면 학습을 멈춤
    early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    # 훈련 중간 중간 현재 Parameter의 값들을 저장
    filename = 'checkpoint-epoch-{}-batch-{}-trial-001.h5'.format(ai_settings['epochs'], ai_settings['batch_size'])
    checkpointer = ModelCheckpoint(filename,  # file명을 지정합니다
                                 monitor='val_loss',  # val_loss 값이 개선되었을때 호출됩니다
                                 verbose=1,  # 로그를 출력합니다
                                 save_best_only=True,  # 가장 best 값만 저장합니다
                                 mode='auto'  # auto는 알아서 best를 찾습니다. min/max
                                 )

    # model 생성
    model = create_model(x_train, ai_settings)
    # model 학습. 휸련데이터샛을 이용해 epochs만큼 반복 훈련 (논문에선 5000으로 설정). verbose 로그 출력 설정
    # validation_data를 총해 에폭이 끝날 때마다 학습 모델을 해당 데이터로 평가한다. 해당 데이터로 학습하지는 않는다.
    # history = model.fit(x_train, y_train, epochs=ai_settings['epochs'], batch_size=ai_settings['batch_size'],
    #                     validation_data=(x_val, y_val), shuffle=False,
    #                     callbacks=[CustomCallback()])
    history = model.fit(x_train, y_train, epochs=ai_settings['epochs'], batch_size=ai_settings['batch_size'],
                        validation_data=(x_val, y_val), shuffle=False,
                        callbacks=[early_stopping, checkpoint])
    fig = plot_model_fit_history(code, history, ai_settings['epochs'])
    fig.show()

    # model 저장
    folder_name = 'models'
    model_fname = f"""{folder_name}/{code}_model_{ai_settings['table']}_epoch_{ai_settings['epochs']}_nstep_{ai_settings['num_step']}\
_units_{ai_settings['num_units']}_batch_{ai_settings['batch_size']}\
_learning_rate_{str(ai_settings['learning_rate']).replace('0.', '')}_optimizer_{ai_settings['optimizer']}\
_loss_{ai_settings['loss']}_activation_{ai_settings['activation']}.h5"""
    model.save(model_fname)
    # 내일 오를지 내릴지에 대한 label 예측 값 출력
    predicted = model.predict(x_test)
    y_pred = np.argmax(predicted, axis=1)
    Y_test = np.argmax(y_test, axis=1)

    back_testing(code, test_sample_df, y_pred, ai_settings, history)

    # 모델 리턴해서 내일 오른다는 예측이 있을 경우 매수 떨어지면 매도
    if y_pred[-1] == 1:
        return False
    else:
        return True


def use_model(ai_filter_num, coin_df, code):
    if ai_filter_num == 1001:
        ai_settings = {
            "table": "min1",
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 10,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "sigmoid",
            "is_continuously_train": False
        }
    else:
        print("ai_filter_num 설정 오류")
        exit(1)

    # test 데이터의 사이즈가 클 경우 예측률 하락
    if len(coin_df) > 100:
        coin_df = coin_df[:100].copy()

    coin_df['next_rtn'] = coin_df['close'] / coin_df['open'] - 1
    default_features = ['open', 'high', 'low', 'close', 'volume']
    # add_features = ['ubb', 'mbb', 'lbb']
    add_features = []
    extra_list = ['next_rtn']
    all_features = default_features + add_features + extra_list

    # 최소-최대 정규화
    coin_sample_df, eng_list = min_max_normal(coin_df, all_features, extra_list)
    # n일 이동평균값 이용 시 nan 값 있는 데이터 제거, nan 값 제거 후 model.fit 할 때 loss도 정상 출력, 에측값도 정상
    nan_cnt = np.where(np.isnan(coin_sample_df))[0][-1] + 1
    train_sample_df = coin_sample_df[nan_cnt:]

    if len(np.where(np.isnan(train_sample_df))[0]) == 0:
        pass
    else:
        nan_cnt = np.where(np.isnan(train_sample_df))[0][-1] + 1
        train_sample_df = train_sample_df[nan_cnt:]

    # 레이블링 테이터
    # (num_step)일치 (n_feature)개 변수의 데이터를 사용해 다음날 종가 예측
    num_step = 5
    eng_list = eng_list + extra_list
    n_feature = len(eng_list) - 1  # next_rtn 을 제외한 feature 개수

    # 훈련, 검증, 테스트 데이터를 변수 데이터와 레이블 데이터로 나눈다
    x_test, y_test = create_dataset_binary(train_sample_df, eng_list, num_step, n_feature)

    # 모델 불러오기
    from tensorflow.keras.models import load_model
    try:
        folder_name = 'models'
        model_fname = f"""{folder_name}/{code}_model_{ai_settings['table']}_epoch_{ai_settings['epochs']}_nstep_{ai_settings['num_step']}\
    _units_{ai_settings['num_units']}_batch_{ai_settings['batch_size']}\
    _learning_rate_{str(ai_settings['learning_rate']).replace('0.', '')}_optimizer_{ai_settings['optimizer']}\
    _loss_{ai_settings['loss']}_activation_{ai_settings['activation']}.h5"""

        model = load_model(model_fname)
    except Exception:
        # 저장된 모델이 없는 경우 학습
        filtered = train_model(ai_filter_num, coin_df, code)

    # 예측
    predicted = model.predict(x_test)
    y_pred = np.argmax(predicted, axis=1)

    return y_pred
