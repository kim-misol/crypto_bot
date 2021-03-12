"""
num_unit : RNN(LSTM) 신경망에 존재하는 뉴런의 개수
epoch : 반복 학습 횟수
batch_size : 데이터셋을 나눠서 학습 (64~512가 대표적, 컴퓨터 메모리가 배치되어 있는 방식과 연관되어 2의 지수값을 가질 때 더 빨리 학습가능하다.)
learning_rate : 학습 속도 감쇠법
optimizer : 최적화 방식
loss : 손실함수
activation : 활성화 함수 설정합니다.
    - ‘linear’ : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.
    - ‘relu’ : rectifier 함수, 은익층에 주로 쓰입니다.
    - ‘sigmoid’ : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다.
    - ‘softmax’ : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다.
"""


def get_ai_settings(ai_filter_num, min_unit):
    # step 5 units 200 epoch 50 batch 10
    if ai_filter_num == 1:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 10,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "sigmoid",
            "is_continuously_train": False
        }
    # step 10 units 200 epoch 50 batch 32
    # step 50 units 200 epoch 50 batch 32

    # step 5 units 50 epoch 50 batch 32
    # step 5 units 1000 epoch 50 batch 32

    # step 5 units 200 epoch 50 batch 32
    elif ai_filter_num == 101:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "sigmoid",
            "is_continuously_train": False
        }
    elif ai_filter_num == 102:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "softmax",
            "is_continuously_train": False
        }
    elif ai_filter_num == 103:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "tanh",
            "is_continuously_train": False
        }
    # 제외) 50, 5, 200, 32, 0.01, adam, categorical_crossentropy, relu loss값이 nan으로 나온다
    elif ai_filter_num == 104:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "relu",
            "is_continuously_train": False
        }
    elif ai_filter_num == 105:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "softsign",
            "is_continuously_train": False
        }
    elif ai_filter_num == 106:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "softplus",
            "is_continuously_train": False
        }
    elif ai_filter_num == 107:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "sigmoid",
            "is_continuously_train": False
        }
    elif ai_filter_num == 108:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "softmax",
            "is_continuously_train": False
        }
    elif ai_filter_num == 109:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "tanh",
            "is_continuously_train": False
        }
    elif ai_filter_num == 110:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "relu",
            "is_continuously_train": False
        }
    elif ai_filter_num == 111:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "softsign",
            "is_continuously_train": False
        }
    elif ai_filter_num == 112:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "softplus",
            "is_continuously_train": False
        }
    elif ai_filter_num == 113:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "sigmoid",
            "is_continuously_train": False
        }
    elif ai_filter_num == 114:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "softmax",
            "is_continuously_train": False
        }
    elif ai_filter_num == 115:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "tanh",
            "is_continuously_train": False
        }
    #  주식에서 정확도 1위
    elif ai_filter_num == 116:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "relu",
            "is_continuously_train": False
        }
    elif ai_filter_num == 117:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "softsign",
            "is_continuously_train": False
        }
    elif ai_filter_num == 118:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "softplus",
            "is_continuously_train": False
        }
    # step 5 units 200 epoch 50 batch 64
    # 코인 10분에서 정확도 1위
    elif ai_filter_num == 120:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "sigmoid",
            "is_continuously_train": False
        }
    elif ai_filter_num == 121:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "softmax",
            "is_continuously_train": False
        }
    elif ai_filter_num == 122:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "tanh",
            "is_continuously_train": False
        }
    elif ai_filter_num == 123:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "softsign",
            "is_continuously_train": False
        }
    elif ai_filter_num == 124:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "softplus",
            "is_continuously_train": False
        }
    elif ai_filter_num == 125:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "sigmoid",
            "is_continuously_train": False
        }
    elif ai_filter_num == 126:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "softmax",
            "is_continuously_train": False
        }
    elif ai_filter_num == 127:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "tanh",
            "is_continuously_train": False
        }
    elif ai_filter_num == 128:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "relu",
            "is_continuously_train": False
        }
    elif ai_filter_num == 129:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "softsign",
            "is_continuously_train": False
        }
    elif ai_filter_num == 130:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "softplus",
            "is_continuously_train": False
        }
    elif ai_filter_num == 131:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "sigmoid",
            "is_continuously_train": False
        }
    elif ai_filter_num == 132:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "softmax",
            "is_continuously_train": False
        }
    elif ai_filter_num == 133:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "tanh",
            "is_continuously_train": False
        }
    elif ai_filter_num == 134:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "relu",
            "is_continuously_train": False
        }
    elif ai_filter_num == 135:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "softsign",
            "is_continuously_train": False
        }
    elif ai_filter_num == 136:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "softplus",
            "is_continuously_train": False
        }
    # @ epoch 400
    elif ai_filter_num == 137:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 400,
            "batch_size": 64,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "sigmoid",
            "is_continuously_train": False
        }
    # @ 3xmin_unit 뒤 예측
    elif ai_filter_num == 1001:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 3,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "sigmoid",
            "is_continuously_train": False
        }
    # @ 3xmin_unit 뒤 예측
    elif ai_filter_num == 1002:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 3,
            "num_step": 5,
            "num_units": 50,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "sigmoid",
            "is_continuously_train": False
        }

    # step 5 units 200 epoch 50 batch 32 learning_rate 0.001
    elif ai_filter_num == 1101:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "sigmoid",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1102:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "softmax",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1103:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "tanh",
            "is_continuously_train": False
        }
    # 제외) 50, 5, 200, 32, 0.01, adam, categorical_crossentropy, relu loss값이 nan으로 나온다
    elif ai_filter_num == 1104:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "relu",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1105:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "softsign",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1106:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "softplus",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1107:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "sigmoid",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1108:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "softmax",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1109:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "tanh",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1110:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "relu",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1111:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "softsign",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1112:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "softplus",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1113:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "sigmoid",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1114:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "softmax",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1115:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "tanh",
            "is_continuously_train": False
        }
    #  주식에서 정확도 1위
    elif ai_filter_num == 1116:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "relu",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1117:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "softsign",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1118:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "softplus",
            "is_continuously_train": False
        }
    # step 5 units 200 epoch 50 batch 64 learning_rate 0.001
    elif ai_filter_num == 1119:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "sigmoid",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1120:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "softmax",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1121:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "tanh",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1122:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "softsign",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1123:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "softplus",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1124:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "sigmoid",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1125:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "softmax",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1126:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "tanh",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1127:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "relu",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1128:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "softsign",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1129:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "mse",
            "activation": "softplus",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1130:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "sigmoid",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1131:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "softmax",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1132:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "tanh",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1133:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "relu",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1134:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "softsign",
            "is_continuously_train": False
        }
    elif ai_filter_num == 1135:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "softplus",
            "is_continuously_train": False
        }

    # 위의 테스트로 얻은 조합
    elif ai_filter_num == 10001:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 50,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "sigmoid",
            "is_continuously_train": False
        }
    elif ai_filter_num == 10002:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 50,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "relu",
            "is_continuously_train": False
        }
    # 테스트용 1
    elif ai_filter_num == 999:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 10,
            "batch_size": 10,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "activation": "sigmoid",
            "is_continuously_train": False
        }
    # 테스트용 2
    elif ai_filter_num == 998:
        ai_settings = {
            "table": f"min{min_unit}",
            "n_pred": 1,
            "num_step": 5,
            "num_units": 200,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "activation": "relu",
            "is_continuously_train": False
        }
    else:
        print("ai_filter_num 설정 오류")
        exit(1)

    return ai_settings
