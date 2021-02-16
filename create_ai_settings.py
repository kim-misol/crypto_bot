import itertools

# s_list = []
simul_num = 'elif ai_filter_num == {i}:\nai_settings = {\n\t'
table = '"table": "min_craw",\n'
num_step = ['"num_step": 5,\n"num_units": 200,\n"epochs": 50,', '"num_step": 100,\n"num_units": 200,\n"epochs": 50,']
batch_size = ['"batch_size": 10,\n"learning_rate": 0.001,\n"optimizer": "adam",', '"batch_size": 32,\n"learning_rate": 0.001,\n"optimizer": "adam",']
loss = ['"loss": "categorical_crossentropy",', '"loss": "mse",', '"loss": "binary_crossentropy",']
activation = ['"activation": "sigmoid",\n"is_continuously_train": False', '"activation": "softmax",\n"is_continuously_train": False', '"activation": "tanh",\n"is_continuously_train": False',
              '"activation": "relu",\n"is_continuously_train": False', '"activation": "softsign",\n"is_continuously_train": False', '"activation": "softplus",\n"is_continuously_train": False']


result = list(itertools.product(*[num_step,batch_size, loss, activation]))
print(len(result))
# print(result)

for r in result:
    for rr in r:
        print(rr)
    print("\n")

