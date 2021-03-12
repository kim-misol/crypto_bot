import itertools


# 초반에 좋은 조합의 모델 학습 조건 찾아보기 위해서
def run_v1():
    simul_num = ['elif ai_filter_num == {i}:\n\tai_settings = {']
    table = ['\t"table":  f"min{min_unit}",']
    num_step = ['\t"num_step": 5,\n\t"num_units": 200,\n\t"epochs": 50,']
    batch_size = ['\t"batch_size": 64,\n\t"learning_rate": 0.001,\n\t"optimizer": "adam",']
    loss = ['\t"loss": "categorical_crossentropy",', '\t"loss": "mse",', '\t"loss": "binary_crossentropy",']
    activation = ['\t"activation": "sigmoid",\n\t"is_continuously_train": False',
                  '\t"activation": "softmax",\n\t"is_continuously_train": False',
                  '\t"activation": "tanh",\n\t"is_continuously_train": False',
                  '\t"activation": "relu",\n\t"is_continuously_train": False',
                  '\t"activation": "softsign",\n\t"is_continuously_train": False',
                  '\t"activation": "softplus",\n\t"is_continuously_train": False']

    result = list(itertools.product(*[simul_num, table, num_step, batch_size, loss, activation, ['}']]))

    for r in result:
        for rr in r:
            print(rr)
    print(len(result))


def run():
    simul_num = ['elif ai_filter_num == {i}:\n\tai_settings = {']
    table = ['\t"table":  f"min{min_unit}",']
    num_step = ['\t"num_step": 5,\n\t"num_units": 50,\n\t"epochs": 50,']
    batch_size = ['\t"batch_size": 32,\n\t"learning_rate": 0.001,\n\t"optimizer": "adam",']
    loss_activation = ['\t"loss": "categorical_crossentropy",\n\t"activation": "sigmoid",\n\t"is_continuously_train": False',
                  '\t"loss": "binary_crossentropy",\n\t"activation": "relu",\n\t"is_continuously_train": False']

    result = list(itertools.product(*[simul_num, table, num_step, batch_size, loss_activation, ['}']]))

    for r in result:
        for rr in r:
            print(rr)
    print(len(result))


if __name__ == "__main__":
    run()
