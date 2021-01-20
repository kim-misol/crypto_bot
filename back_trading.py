import os
from datetime import datetime, timedelta
from pathlib import Path

import FinanceDataReader as fdr
import backtrader as bt
import numpy as np
import pandas_datareader as pdr

from graphs import draw_candle, draw_candle_with_indicator, plot_model_fit_history
from data.code_list import stock_codes, coin_codes
from trading_indicators import bollinger_band
from ai import data_split, min_max_normal, create_dataset_binary, create_model


class TestStrategy(bt.Strategy):

    def __init__(self):
        '''
        self.datas[0]을 통해 앞서
        dts = bt.feeds.PandasData(dataname=stockData,open='Open',close='Close')
        에서 전달한 open, close를 불러올 수 있다.
        ex) self.datas[0].open

        데이터필드 확인 방법
        self.datas[0].datafields
        ['datetime', 'open', 'high', 'low', 'close', 'volume', 'openinterest']
        '''
        # index 0에 오늘 값 1에 가장오래된 값 -1에 어제 값이 들어있다
        self.close = self.datas[0].close
        self.date = self.datas[0].datetime
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.i = 0
        self.pastMA5 = 0
        self.pastMA20 = 0
        self.meanPrice = 0
        self.numberOfStocks = 0
        self.simul_num = 2

    '''
    Buy나 Sell 등 오더가 일어났을 때 호출되는 메서드이다
    전달값인 order의 다양한 attributes를 알고 싶다면 아래 링크 Documentation에서 확인할 수 있다. ex) order.size, order.price
    https://www.backtrader.com/docu/order/#order-notification
    '''

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        elif order.status in [order.Completed]:
            if order.isbuy():
                self.meanPrice = (self.meanPrice * self.numberOfStocks + order.size * self.close[0]) / (
                        self.numberOfStocks + order.size)
                self.numberOfStocks += order.size
            elif order.issell():
                self.meanPrice = 0
                self.numberOfStocks = 0
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print('Order Canceled/Margin/Rejected')
        self.order = None

    '''
    거래가 일어났을 때 호출되는 메서드이다.
    trade.isclosed가 True일 때(Trade가 종료되었을 때) profit을 나타내는 코드이다
    '''

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        # print(f'{self.datas[0].datetime.date(0)} PROFIT, GROSS {trade.pnl}, NET {trade.pnlcomm}')

    '''
    실제 백테스팅을 하는 부분이다. 
    하루하루 지날때 마다 next 메서드가 한 번씩 호출된다. 
    self.position이 True면 주식을 보유하고 있다는 의미이다. 
    또한 self.dataclose[0]는 현재 시뮬레이션 상 날짜(0)로, next메서드가 호출될 때 마다 한 칸씩 뒤로 밀리게 된다
    즉, 언제나 당일을 나타내는 index는 0이다
    
    여기서 backtrader의 특징 중 하나가 나오는데, 전 날, 전전날을 나타내는 index는 1이나 2가 아니라, -1과 -2라는 점이다. 
    
    주문을 하기 위해선 self.order = self.buy() 혹은 self.sell()을 하면 된다. 
    () 안에는 주문할 물량을 전달해 주면 된다.
    '''

    def next(self):
        self.i += 1
        if self.i < 20:
            return
        else:
            MA5 = np.mean([self.close[n - 4] for n in range(5)])
            MA20 = np.mean([self.close[n - 19] for n in range(20)])
            if self.simul_num == 1:
                # 매수 조건
                if MA5 > MA20 and self.pastMA5 < self.pastMA20 and self.broker.getcash() > self.close[0]:
                    amountToOrder = int(self.broker.getcash() / self.close[0])
                    self.order = self.buy(size=amountToOrder)
                    print(f"{self.datas[0].datetime.date(0)} - BUY : {self.close[0]}, buy amount : {amountToOrder}")
                # 매도 조건
                elif self.meanPrice * 1.1 < self.close[0] and self.numberOfStocks != 0:
                    print(
                        f"{self.datas[0].datetime.date(0)} - SELL : {self.close[0]}, sell amount : {self.numberOfStocks}")
                    self.order = self.sell(size=self.numberOfStocks)
            # 5 20 골든크로스 데드크로스
            elif self.simul_num == 2:
                # 매수 조건
                if MA5 > MA20 and self.pastMA5 < self.pastMA20 and self.broker.getcash() > self.close[0]:
                    amountToOrder = int(self.broker.getcash() / self.close[0])
                    self.order = self.buy(size=amountToOrder)
                    print(f"{self.datas[0].datetime.date(0)} - BUY : {self.close[0]}, buy amount : {amountToOrder}")
                # 매도 조건
                elif MA5 > MA20 and self.pastMA5 < self.pastMA20 and self.numberOfStocks != 0:
                    print(
                        f"{self.datas[0].datetime.date(0)} - SELL : {self.close[0]}, sell amount : {self.numberOfStocks}")
                    self.order = self.sell(size=self.numberOfStocks)

            # 과거 5,20일 이동 평균값을 저장해둔다
            self.pastMA5 = MA5
            self.pastMA20 = MA20


def data_settings(code, start=datetime(2020, 1, 1), end=datetime.today()):
    # 비트코인 원화 가격 (빗썸) 2016년~현재
    price_df = fdr.DataReader(code, start=start, end=end)
    # 결측치 존재 유무 확인
    invalid_data_cnt = len(price_df[price_df.isin([np.nan, np.inf, -np.inf]).any(1)])

    if invalid_data_cnt == 0:
        price_df['Date'] = price_df.index
        price_df['Open'] = price_df.iloc[:]['Open'].astype(np.float64)
        price_df['High'] = price_df.iloc[:]['High'].astype(np.float64)
        price_df['Low'] = price_df.iloc[:]['Low'].astype(np.float64)
        price_df['Close'] = price_df.iloc[:]['Close'].astype(np.float64)
        price_df['Volume'] = price_df.iloc[:]['Volume']
        price_df = bollinger_band(price_df)
        df = price_df.loc[:, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ubb', 'mbb', 'lbb']].copy()
        return df
    return False


def simulator():
    '''
    다음으로는 backtrader의 몸통 역할을 하는 Cerebro() 객체를 만들어준다.
    이렇게 만들어진 cerebro에는 cerebro.broker 클래스를 이용해
    초기자금 설정(setcash()), 수수료 설정(setcommission(), commission=0.001은 0.1%수수료라는 뜻)
    등의 초기 설정을 할 수 있다.
    '''
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(1000000.0)
    # cerebro.broker.setcommission(commission=0.001)
    # 주식
    cerebro.broker.setcommission(commission=0.0033)

    '''
    다음으론 백테스팅에 사용할 알고리즘을 addstrategy()를 이용해 넣는다. 
    여기에 넣은 TestStrategy는 Strategy 클래스를 상속받아 만든 커스텀 클래스이다.
    '''
    cerebro.addstrategy(TestStrategy)

    '''
    backtrader에 사용되는 데이터는 backtrader의 자체 데이터 구조를 사용한다. 
    이러한 구조로 만들기 위해 자체적으로 제공하는 backtester.feeds 클래스를 사용한다. 
    이 클래스 내에선 CSV, Pandas 등의 데이터 구조를 backtrader에 맞는 데이터 구조로 변환해주는 메서드를 제공한다. 
    pandas dataframe을 사용했으므로, bt.feeds.PandasData를 사용해 dataname, open, close, volume, high와 같은 값들을 전달한다. 
    전달된 값은 TestStrategy와 같은 백테스팅 알고리즘 내에서 사용될 수 있다. 

    만약 open값을 전달하지 않으면 
    알고리즘 실행 후 cerebro.broker.getvalue()를 통해 포트폴리오 가치를 불러올 때 Nan이 리턴되므로, 
    open값은 꼭 전달하도록 한다.
    이렇게 데이터 생성이 끝나면, adddata()를 통해 데이터를 투입해준다.
    '''
    dataset = bt.feeds.PandasData(dataname=coin_df, open='Open', close='Close')
    cerebro.adddata(dataset)

    # 백테스팅 전 초기 자금 출력
    deposit = cerebro.broker.getvalue()
    print(f'Starting portfolio Value : {round(cerebro.broker.getvalue(), 2)}')
    # cerebro.run()에서 백테스팅 알고리즘을 실행
    cerebro.run()

    # 백테스팅 종료 후 cerebro.broker.getvalue()를 실행시키면 알고리즘 시뮬 결과 출력
    profit = round(cerebro.broker.getvalue() - deposit, 2)
    rate = round(cerebro.broker.getvalue() / deposit * 100, 2)
    backtesting_result = f"""
    Final portfolio value : {round(cerebro.broker.getvalue(), 2)}
    Total Profit: {profit}
    Rate: {rate}%
    """
    print(backtesting_result)
    return profit, rate


def save_graph(coin_df, code):
    # $ 그래프 그리기
    # draw_candle(coin_df, code)
    fig = draw_candle_with_indicator(coin_df, code)
    today = str(datetime.today())[:10].replace('-', '')
    fcode = code.replace('/', '-')

    if not os.path.exists("images"):
        os.mkdir("images")
    if not os.path.exists(f"images/{today}"):
        os.mkdir(f"images/{today}")

    fig.write_image(f"images/{today}/{fcode}.png")
    fig.write_html(f"images/{today}/{fcode}.html")


def ai_filter(coin_df):
    # 학습, 검증, 테스트 데이터 기간 분할 6:2:2
    train_df, val_df, test_df = data_split(coin_df)
    # 최소-최대 정규화
    train_sample_df, eng_list = min_max_normal(train_df)
    # n일 이동평균값 이용 시 nan 값 있는 데이터 제거, nan 값 제거 후 model.fit 할 때 loss도 정상 출력, 에측값도 정상
    nan_cnt = np.where(np.isnan(train_sample_df))[0][-1] + 1
    train_sample_df = train_sample_df[nan_cnt:]
    val_sample_df, eng_list = min_max_normal(val_df)
    test_sample_df, eng_list = min_max_normal(test_df)
    # 레이블링 테이터
    # (num_step)일치 (n_feature)개 변수의 데이터를 사용해 다음날 종가 예측
    num_step = 5
    num_unit = 200
    n_feature = len(eng_list)

    # 훈련, 검증, 테스트 데이터를 변수 데이터와 레이블 데이터로 나눈다
    x_train, y_train = create_dataset_binary(train_sample_df, eng_list, num_step, n_feature)
    x_val, y_val = create_dataset_binary(val_sample_df, eng_list, num_step, n_feature)
    x_test, y_test = create_dataset_binary(test_sample_df, eng_list, num_step, n_feature)

    # model 생성
    model = create_model(x_train, num_unit)
    # ? 모델 저장하여 재사용 가능한지 확인
    # model 학습. 휸련데이터샛을 이용해 epochs만큼 반복 훈련 (논문에선 5000으로 설정). verbose 로그 출력 설정
    # validation_data를 총해 에폭이 끝날 때마다 학습 모델을 해당 데이터로 평가한다. 해당 데이터로 학습하지는 않는다.
    EPOCHS = 20
    BATCH_SIZE = 10
    VERBOSE = 1
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE,
                        validation_data=(x_val, y_val))
    # fig = plot_model_fit_history(history, EPOCHS)
    # fig.show()
    # 예측 값 출력
    predicted = model.predict(x_test)
    y_pred = np.argmax(predicted, axis=1)
    Y_test = np.argmax(y_test, axis=1)
    # cm = confusion_matrix(Y_test, y_pred)
    # report = classification_report(Y_test, y_pred)


if __name__ == "__main__":
    # 주식 또는 가상암호화폐 종목 코드 리스트 가져오기
    # $ 코드 리스트
    # code_list = stock_codes[:2]
    code_list = coin_codes
    total_profit, sum_rate = 0, 0
    # use_graph = True if input(f"그래프 저장 여부 : (y or n) ") in ('Y', 'y', 1, 'ㅛ') else False
    # use_ai = True if input(f"AI 학습 사용 여부 : (y or n) ") in ('Y', 'y', 1, 'ㅛ') else False
    use_graph = False
    use_ai = True
    if use_graph not in ('Y', 'y', 1, 'ㅛ', 'N', 'n', 0, 'ㅜ') or use_ai not in ('Y', 'y', 1, 'ㅛ', 'N', 'n', 0, 'ㅜ'):
        print('y 또는 n을 입력해주세요.')
        exit(1)

    for code in code_list:
        print(f"종목 코드: {code}")
        get_data_start = datetime.now()
        # 종목별 데이터
        # $ 백테스팅 시작 날짜 설정
        coin_df = data_settings(code=code, start=datetime(2018, 1, 1))
        # get_data_time = datetime.now() - get_data_start
        # sum_get_data_time += get_data_time

        if coin_df is not False:
            # ai 학습을 활용할 경우
            if use_ai:
                ai_filter(coin_df)

            # 해당 종목 시뮬
            profit, rate = simulator()
            # 손익 계산
            total_profit += profit
            sum_rate += rate
            print(f"현재까지의 수익: {total_profit}\n")
            if use_graph:
                save_graph(coin_df, code)
        else:
            print('데이터에 결측치가 존재합니다.')

    total_rate = sum_rate / len(code_list)
    print(f"총 수익: {total_profit}\n수익률: {total_rate}")
    # print(f"시뮬 종료: {datetime.now()}\n소요 시간: {datetime.now() - simul_start}")
    # print(f"데이터 콜렉팅을 제외한 소요 시간: {datetime.now() - simul_start - sum_get_data_time}")
