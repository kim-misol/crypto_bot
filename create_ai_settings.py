import itertools

# s_list = []
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
