'''
history direactory에 쌓인 모델별, 종목별 예측 확률 비교
'''
import ast
from datetime import datetime
from pathlib import Path

import pandas as pd
from tensorflow.keras.models import load_model
from library.ai_model import data_split, min_max_normal, create_dataset_binary


def save_hyperparams_n_results_from_history():
    dir_name = 'history'
    cwd = Path.cwd() / dir_name
    files = list(cwd.glob('*'))
    print(files)
    df = pd.DataFrame(columns=['market', 'code', 'model_fname', 'start_date', 'train_val_train', 'table_unit',
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

        c = pd.DataFrame(
            [['crypto', code, fname, "2020-01-01 01:00:00", "6:2:2", ai['table'], ai['epochs'], ai['num_step'],
              ai['num_units'],
              ai['batch_size'], ai['learning_rate'], ai['optimizer'], ai['loss'], ai['activation'],
              avg_loss, avg_acc, avg_val_loss, avg_val_acc, last_val_acc, max_val_acc, min_val_acc, val_acc_list]],
            columns=['market', 'code', 'model_fname', 'start_date', 'train_val_train', 'table_unit',
                     'epoch', 'ntep', 'units', 'batch', 'learning_rate', 'optimizer',
                     'loss', 'activation', 'avg_loss', 'avg_acc', 'avg_val_loss',
                     'avg_val_acc', 'last_val_acc', 'max_val_acc', 'min_val_acc', 'val_acc_list'])
        df = df.append(c)
    df.to_excel(f"train_log/results_{str(datetime.now())[:16].replace(':', '_')}.xlsx")


def get_test_data(coin_df, ai):
    coin_df['next_rtn'] = coin_df['close'] / coin_df['open'] - 1
    # 학습, 검증, 테스트 데이터 기간 분할 6:2:2
    train_df, val_df, test_df = data_split(coin_df)
    default_features = ['open', 'high', 'low', 'close', 'volume']
    add_features = []
    extra_list = ['next_rtn']
    all_features = default_features + add_features + extra_list

    # 최소-최대 정규화
    test_sample_df, eng_list = min_max_normal(test_df, all_features, extra_list)

    # 레이블링 테이터
    eng_list = eng_list + extra_list
    n_feature = len(eng_list) - 1  # next_rtn 을 제외한 feature 개수

    # 훈련, 검증, 테스트 데이터를 변수 데이터와 레이블 데이터로 나눈다
    x_test, y_test = create_dataset_binary(test_sample_df, eng_list, ai, n_feature)
    return x_test, y_test


def save_hyperparams_n_results_from_model():
    date_start = 2020
    dir_name = f'history/{date_start}'
    cwd = Path.cwd() / dir_name
    files = list(cwd.glob('*'))
    df = pd.DataFrame(
        columns=['market', 'code', 'model_fname', 'start_date', "accuracy", "n_pred", 'table_unit', 'epoch',
                 'ntep', 'units', 'batch', 'learning_rate', 'optimizer', 'loss',
                 'activation', 'avg_loss', 'avg_acc', 'avg_val_loss', 'avg_val_acc', 'last_val_acc', 'max_val_acc',
                 'min_val_acc', 'val_acc_list', 'train_val_train'])

    for file in files:
        fname = str(file).split('\\')[-1][:-4]
        code = fname.split('_')[0]
        # 인코딩 에러 발생하는 경우
        try:
            text = open(file, 'rt', encoding='UTF8').read()
        except Exception:
            text = file.read_text()

        txt_list = text.split("\n")
        ai_str = txt_list[-1]
        ai = ast.literal_eval(ai_str)

        avg_loss = text.split("mean loss:")[1].split("\n")[1]
        avg_acc = text.split("mean acc:")[1].split("\n")[1]
        avg_val_loss = text.split("mean val_loss:")[1].split("\n")[1]
        avg_val_acc = text.split("mean val_acc:")[1].split("\n")[1]
        last_val_acc = text.split("last val_acc:")[1].split("\n")[1]
        max_val_acc = text.split("MAX:")[1].split("\n")[1]
        min_val_acc = text.split("MIN:")[1].split("\n")[1]
        val_acc_list = text.split("MIN:")[1].split("\n")[2]

        unit = ai['table'][3:]

        if '모델 정확도' in text:
            accuracy = f"{round(float(txt_list[-2][:-1]), 2)}%"
        else:
            try:
                from back_trading import data_settings
                if unit in (1, 3):
                    if code == "KRW-BTC":
                        market = 1
                    else:
                        market = 44
                else:
                    market = code
                # 모델 정확도 평가를 위한 데이터 전처리
                coin_df = data_settings(market=market, unit=unit, date_start=date_start)
                x_test, y_test = get_test_data(coin_df, ai)

                # 모델 불러오기
                model_fname = f"models/{fname}.h5"
                model = load_model(model_fname)
                # for i in range(10):
                loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
                accuracy = f"{round(accuracy * 100, 2)}%"
            except Exception:
                accuracy = "0"
        print(f"{fname}: {accuracy}")

        c = pd.DataFrame(
            [['crypto', code, fname, "2020-01-01 01:00:00", accuracy, ai['n_pred'], ai['table'], ai['epochs'],
              ai['num_step'], ai['num_units'], ai['batch_size'], ai['learning_rate'], ai['optimizer'], ai['loss'],
              ai['activation'], avg_loss, avg_acc, avg_val_loss, avg_val_acc, last_val_acc, max_val_acc,
              min_val_acc, val_acc_list, "6:2:2"]],
            columns=['market', 'code', 'model_fname', 'start_date', "accuracy", "n_pred", 'table_unit', 'epoch',
                     'ntep', 'units', 'batch', 'learning_rate', 'optimizer', 'loss',
                     'activation', 'avg_loss', 'avg_acc', 'avg_val_loss', 'avg_val_acc', 'last_val_acc', 'max_val_acc',
                     'min_val_acc', 'val_acc_list', 'train_val_train'])
        df = df.append(c)
    df.to_excel(f"train_log/results_{str(datetime.now())[:16].replace(':', '_')}.xlsx")


# train log에서 val_accuracy 리스트로 만들어서 예측률 비교
def get_val_accuracy():
    dir_name = 'train_log'

    cwd = Path.cwd() / dir_name
    files = list(cwd.glob('*'))

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


def edit_history():
    dir_name = 'history/2020'
    from_text = ", 'num_step':"
    to_text = ",'n_pred': 1, 'num_step':"

    cwd = Path.cwd() / dir_name
    files = list(cwd.glob('*'))

    for file in files:
        try:
            text = open(file, 'rt', encoding='UTF8').read()
        except Exception:
            text = file.read_text()

        try:
            text = text.replace(from_text, to_text)
            encoded_data = text.encode("utf8")
            f = open(file, 'wb')
            f.write(encoded_data)
            f.close()
        except Exception:
            print(file)
            print(text)


def change_filename():
    import os
    files = path_setting()
    # 파일명 변경 코드
    for file in files:
        fname = str(file)
        # new_fname = fname.replace('__', '_')
        new_fname = fname.replace('001', '01')
        # new_fname = fname.replace('epoch', 'epoch_')
        # new_fname = new_fname.replace('nstep', 'nstep_')
        # new_fname = new_fname.replace('units', 'units_')
        # new_fname = new_fname.replace('batch', 'batch_')
        # new_fname = new_fname.replace('rate', 'rate_')
        # new_fname = new_fname.replace('optimizer', 'optimizer_')
        # new_fname = new_fname.replace('loss', 'loss_')
        # new_fname = new_fname.replace('activation', 'activation_')
        print(new_fname)
        os.rename(fname, new_fname)


def path_setting():
    # dir_name = 'models/2020'
    dir_name = 'history/2020'
    cwd = Path.cwd() / dir_name
    print(cwd)
    files = list(cwd.glob('*'))
    print(files)
    return files


if __name__ == "__main__":
    # get_val_accuracy()
    # history로 저장된 텍스트 파일의 내용을 가져와서 엑셀에 정리
    # save_hyperparams_n_results_from_history()
    # 파일명 변경
    # change_filename()
    # history 파일 수정
    # edit_history()
    # + model 정확도 추가
    save_hyperparams_n_results_from_model()
