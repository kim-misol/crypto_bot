'''
history direactory에 쌓인 모델별, 종목별 예측 확률 비교
'''
import os
from pathlib import Path
from datetime import datetime
import statistics
import pandas as pd
import ast

def save_model_hyperparams_results():
    dir_name = 'history'
    cwd = Path.cwd() / dir_name
    print(cwd)
    files = list(cwd.glob('*'))
    print(files)

    df = pd.DataFrame(columns=['code', 'model_fname', 'start_date', 'train_val_train', 'table_unit',
                              'epoch', 'ntep', 'units', 'batch', 'learning_rate', 'optimizer',
                              'loss', 'activation', 'avg_loss', 'avg_acc', 'avg_val_loss',
                              'avg_val_acc', 'last_val_acc', 'max_val_acc', 'min_val_acc', 'val_acc_list'])
    for file in files:
        fname = str(file).split("\\")[-1][:-4]
        code = fname.split('_')[0]

        text = file.read_text()
        txt_list = text.split("\n")
        ai_str = txt_list[-1]
        ai = ast.literal_eval(ai_str)

        avg_loss = txt_list[16]
        avg_acc = txt_list[20]
        avg_val_loss = txt_list[24]
        avg_val_acc = txt_list[28]
        last_val_acc = txt_list[30]
        max_val_acc = txt_list[32]
        min_val_acc = txt_list[34]
        val_acc_list = txt_list[35]

        c = pd.DataFrame([[code, fname, "2019-01-01 01:00:00", "6:2:2", ai['table'], ai['epochs'], ai['num_step'], ai['num_units'],
                            ai['batch_size'], ai['learning_rate'], ai['optimizer'], ai['loss'], ai['activation'],
                            avg_loss, avg_acc, avg_val_loss, avg_val_acc, last_val_acc, max_val_acc, min_val_acc, val_acc_list]],
                          columns=['code', 'model_fname', 'start_date', 'train_val_train', 'table_unit',
                                   'epoch', 'ntep', 'units', 'batch', 'learning_rate', 'optimizer',
                                   'loss', 'activation', 'avg_loss', 'avg_acc', 'avg_val_loss',
                                   'avg_val_acc', 'last_val_acc', 'max_val_acc', 'min_val_acc', 'val_acc_list'])
        df = df.append(c)
    df.to_excel(f"train_log/results_{str(datetime.now())[:16].replace(':', '_')}.xlsx")



# train log에서 val_accuracy 리스트로 만들어서 예측률 비교
def get_val_accuracy():
    dir_name = 'train_log'
    param = '001'

    cwd = Path.cwd() / dir_name
    print(cwd)
    files = list(cwd.glob('*'))
    print(files)

    # 특정 종목의 모델 학습 변수 별 예측률 비교
    for file in files:
        fname = str(file)
        text = file.read_text()
        t = text.split('val_accuracy: ')[1:]
        val_acc_list = []
        for line in t:
            val_acc = line[:6]
            val_acc_list.append(float(val_acc))
        print(fname)
        print(round(sum(val_acc_list) / len(val_acc_list), 4))


if __name__ == "__main__":
    # get_val_accuracy()
    save_model_hyperparams_results()
