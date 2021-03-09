def get_ai_settings(ai_filter_num, unit):
    # step 5 units 200 epoch 50 batch 10
    if ai_filter_num == 1:
        ai_settings = {
            "table": f"min{unit}",
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
    # step 10 units 200 epoch 50 batch 32
    # step 50 units 200 epoch 50 batch 32

    # step 5 units 50 epoch 50 batch 32
    # step 5 units 1000 epoch 50 batch 32

    # step 5 units 200 epoch 50 batch 32
    elif ai_filter_num == 101:
        ai_settings = {
            "table": f"min{unit}",
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
    elif ai_filter_num == 102:
        ai_settings = {
            "table": f"min{unit}",
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
    elif ai_filter_num == 103:
        ai_settings = {
            "table": f"min{unit}",
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
    elif ai_filter_num == 104:
        ai_settings = {
            "table": f"min{unit}",
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
    elif ai_filter_num == 105:
        ai_settings = {
            "table": f"min{unit}",
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
    elif ai_filter_num == 106:
        ai_settings = {
            "table": f"min{unit}",
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
    elif ai_filter_num == 107:
        ai_settings = {
            "table": f"min{unit}",
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
    elif ai_filter_num == 108:
        ai_settings = {
            "table": f"min{unit}",
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
    elif ai_filter_num == 109:
        ai_settings = {
            "table": f"min{unit}",
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
    elif ai_filter_num == 110:
        ai_settings = {
            "table": f"min{unit}",
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
    elif ai_filter_num == 111:
        ai_settings = {
            "table": f"min{unit}",
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
    elif ai_filter_num == 112:
        ai_settings = {
            "table": f"min{unit}",
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
    elif ai_filter_num == 113:
        ai_settings = {
            "table": f"min{unit}",
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
    elif ai_filter_num == 114:
        ai_settings = {
            "table": f"min{unit}",
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
    elif ai_filter_num == 115:
        ai_settings = {
            "table": f"min{unit}",
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
    elif ai_filter_num == 116:
        ai_settings = {
            "table": f"min{unit}",
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
    elif ai_filter_num == 117:
        ai_settings = {
            "table": f"min{unit}",
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
    elif ai_filter_num == 118:
        ai_settings = {
            "table": f"min{unit}",
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
    # 테스트용 1
    elif ai_filter_num == 999:
        ai_settings = {
            "table": f"min{unit}",
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
    # 테스트용 2
    elif ai_filter_num == 998:
        ai_settings = {
            "table": f"min{unit}",
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
    else:
        print("ai_filter_num 설정 오류")
        exit(1)

    return ai_settings
