from datetime import datetime

import numpy as np

from data.code_list import coin_codes
from library.ai_filter import train_model, use_model
from library.ai_filter_v2 import filter_by_lstm_model
from library.db_conn import get_min_candle
from library.graphs import draw_candle_with_indicator
from library.logging_pack import *
from library.trading_indicators import bollinger_band


def data_settings(market, unit, date_start):
    price_df = get_min_candle(market=market, unit=unit, date_start=date_start)
    # 결측치 존재 유무 확인
    invalid_data_cnt = len(price_df[price_df.isin([np.nan, np.inf, -np.inf]).any(1)])

    if invalid_data_cnt == 0:
        price_df['open'] = price_df.iloc[:]['opening_price'].astype(np.float64)
        price_df['high'] = price_df.iloc[:]['high_price'].astype(np.float64)
        price_df['low'] = price_df.iloc[:]['low_price'].astype(np.float64)
        price_df['close'] = price_df.iloc[:]['trade_price'].astype(np.float64)
        price_df['volume'] = price_df.iloc[:]['acc_trade_volume']
        price_df = bollinger_band(price_df)
        df = price_df.loc[:, ['open', 'high', 'low', 'close', 'volume', 'ubb', 'mbb', 'lbb']].copy()
        return df
    return False


def save_graph(coin_df, code):
    # $ 그래프 그리기
    # draw_candle(coin_df, code)
    fig = draw_candle_with_indicator(coin_df, code)
    today = str(datetime.today())[:10].replace('-', '')
    fcode = code.replace('/', '-')

    if not os.path.exists("img"):
        os.mkdir("img")
    if not os.path.exists(f"img/{today}"):
        os.mkdir(f"img/{today}")

    fig.write_image(f"img/{today}/{fcode}.png")
    fig.write_html(f"img/{today}/{fcode}.html")


def run():
    # 주식 또는 가상암호화폐 종목 코드 리스트 가져오기
    # code_list = coin_codes
    # total_profit, sum_rate = 0, 0
    # use_graph = True if input(f"그래프 저장 여부 : (y or n) ") == 'y' else False
    use_graph = False
    # if use_graph not in ('y', 'n') or use_ai_filter not in ('y', 'n'):
    #     print('y 또는 n을 입력해주세요.')
    #     exit(1)
    # code = input(f"종목코드 입력: (예시. KRW-BTC or KRW-LTC)")
    code = 'KRW-BTC'
    if code == 'KRW-BTC':
        market_id = 1
    elif code == 'KRW-LTC':
        market_id = 44
    else:
        market_id = int(input(f"종목 아이디: (예시. 1 or 44)"))
    date_start = '2020'
    unit = int(input(f"unit: (예시. 1: 1분봉, 3: 3분봉)"))
    if unit in (1, 3):
        market = market_id
    else:
        market = code

    logger.debug(f"종목 코드: {code} {market_id}")
    simul_start = datetime.now()
    # 종목별 데이터
    # $ 백테스팅 시작 날짜 설정
    coin_df = data_settings(market=market, unit=unit, date_start=date_start)

    if coin_df is not False:
        # ai model 학습 또는 사용
        # use_ai_filter = True if input(f"AI 필터 사용 여부 : (y or n) ") == 'y' else False
        # train_ai_model = True if input(f"AI model 학습 여부 : (y or n) ") == 'y' else False
        # ai_filter_num = int(input(f"ai_filter_num 입력: "))
        use_ai_filter = True
        train_ai_model = True
        ai_filter_num = 997

        label = 1  # 1: 산다
        if use_ai_filter:
            if train_ai_model:
                # label = train_model(ai_filter_num, coin_df, code, unit, date_start)
                label = filter_by_lstm_model(ai_filter_num, coin_df, code, unit, date_start)
            else:
                label = use_model(coin_df, code)

        if use_graph:
            save_graph(coin_df, code)
    else:
        logger.debug('데이터에 결측치가 존재합니다.')

    # total_rate = sum_rate / len(code_list)
    # logger.debug(f"총 수익: {total_profit}\n수익률: {total_rate}")
    logger.debug(f"시뮬 종료: {datetime.now()}\n소요 시간: {datetime.now() - simul_start}")


def run_unit_list():
    code = 'KRW-BTC'
    market_id = 1

    # model 학습 시 이용되는 sample data 시작 날짜 설정
    date_start = '2020'
    # units = [60, 30, 15, 10, 5, 3, 1]
    min_unit_list = [10]
    ai_list = list(range(126, 138)) + list(range(1101, 1136))
    # UNIT 3의 ai_filter_num 101 돌려야됨  # unit 별 모델 트레이닝
    for min_unit in min_unit_list:
        if min_unit in (1, 3):
            market = market_id
        else:
            market = code
        # ai setting 별 모델 트레이닝
        for ai_filter_num in ai_list:
            logger.debug(f"종목 코드: {code} {market_id} min_unit:{min_unit} ai_filter_num:{ai_filter_num}")
            simul_start = datetime.now()
            # 종목별 데이터, 백테스팅 시작 날짜 설정
            coin_df = data_settings(market=market, unit=min_unit, date_start=date_start)

            if coin_df is not False:
                # ai model 학습 또는 사용
                label = train_model(ai_filter_num, coin_df, code, min_unit, date_start)
            else:
                logger.debug('데이터에 결측치가 존재합니다.')
            logger.debug(f"시뮬 종료: {datetime.now()}\n소요 시간: {datetime.now() - simul_start}")
    logger.debug(f"최종 시뮬 종료: {datetime.now()}\n소요 시간: {datetime.now() - simul_start}")


def run_start_unit_list():
    # code = input(f"종목코드 입력: (예시. KRW-BTC or KRW-LTC)")
    code = 'KRW-BTC'
    market_id = 1
    # model 학습 시 이용되는 sample data 시작 날짜 설정
    date_start_list = ['2020', '2019', '2018']
    min_unit_list = [10, 3, 1]
    ai_list = [1002, 1001] + list(range(10001, 10003)) + list(range(1110, 1136))
    # 데이터 시작 날짜별 트레이닝
    for date_start in date_start_list:
        # unit 별 모델 트레이닝
        for min_unit in min_unit_list:
            if min_unit in (1, 3):
                market = market_id
            else:
                market = code
            # ai setting 별 모델 트레이닝
            for ai_filter_num in ai_list:
                logger.debug(f"종목: {code} {market_id} min_unit:{min_unit} ai_filter_num:{ai_filter_num} {date_start}부터")
                simul_start = datetime.now()
                # 종목별 데이터, 백테스팅 시작 날짜 설정
                coin_df = data_settings(market=market, unit=min_unit, date_start=date_start)

                if coin_df is not False:
                    # ai model 학습 또는 사용
                    label = train_model(ai_filter_num, coin_df, code, min_unit, date_start)
                else:
                    logger.debug('데이터에 결측치가 존재합니다.')
                logger.debug(f"시뮬 종료: {datetime.now()}\n소요 시간: {datetime.now() - simul_start}")
    logger.debug(f"최종 시뮬 종료: {datetime.now()}\n소요 시간: {datetime.now() - simul_start}")


if __name__ == "__main__":
    run()
    # run_unit_list()
    # run_start_unit_list()
