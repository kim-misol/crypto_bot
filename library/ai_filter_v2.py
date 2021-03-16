import numpy as np
from library.graphs import plot_model_fit_history
from tensorflow.keras.models import load_model

from library.ai_model_v2 import load_data, load_data_v2, train_model, create_model, back_testing
from library.ai_setting_list import get_ai_settings
from library.logging_pack import *


def filter_by_lstm_model(ai_filter_num, df, code, min_unit, date_start):
    # ai_filter_num에 따른 ai setting 가져오기
    ai_settings = get_ai_settings(ai_filter_num, min_unit)
    folder_name = 'models'
    model_fname = f"""{folder_name}/{date_start}/{code}_{ai_settings['table']}_n_pred_{ai_settings['n_pred']}\
_epoch_{ai_settings['epochs']}_nstep_{ai_settings['num_step']}_units_{ai_settings['num_units']}_batch_{ai_settings['batch_size']}\
_learning_rate_{str(ai_settings['learning_rate']).replace('0.', '')}_optimizer_{ai_settings['optimizer']}\
_loss_{ai_settings['loss']}_activation_{ai_settings['activation']}.h5"""
    try:
        # 모델이 존재하는 경우 바로 불러와서 사용
        model = load_model(model_fname)
        return True
    except Exception:
        # 모델이 존재하지 않는 경우 학습
        # 스케일링 and "X_train", "X_test", "y_train", "y_test" 추출
        # dataset = load_data(df=df.copy(), ai_settings=ai_settings)
        dataset = load_data_v2(df=df.copy(), ai_settings=ai_settings)
        # model 생성
        model, callback_list = create_model(dataset, ai_settings, code)
        # model 학습
        model, history = train_model(dataset, model, ai_settings, callback_list)
        # model 저장
        model.save(model_fname)

        # label에 대한 예측 값 출력
        predicted = model.predict(dataset['x_test'])
        y_pred = np.argmax(predicted, axis=1)
        Y_test = np.argmax(dataset['y_test'], axis=1)

        # model 평가
        loss, acc = model.evaluate(dataset['x_test'], dataset['y_test'], verbose=0)
        logger.debug(f"모델 정확도: {float(acc) * 100}%")
        logger.debug(f"1 in y_pred: {1 in y_pred}, 0 in y_pred: {0 in y_pred}")

        back_testing(code, ai_settings, history, acc, date_start)

        # 모델 리턴해서 내일 오른다는 예측이 있을 경우 매수 떨어지면 매도
        if y_pred[-1] == 1:
            return False
        else:
            return True
