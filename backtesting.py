import FinanceDataReader as fdr
import backtrader as bt
import numpy as np


# import matplotlib.pyplot as plt
# %matplotlib inline

# 상위 Strategy를 상속받아 구현
class MyStrategy(bt.Strategy):
    def __init__(self):
        # 간단한 이동평균선 정의 (15일 이동평균) 책 출간 이후 함수명이 변경된 듯 싶다
        # self.sma = bt.indicators.SimpleMovingAverage(period=15)
        self.sma = bt.indicators.MovingAverageSimple(period=15)

    # 매수 매도 알고리즘
    def next(self):
        # 종가가 이동평균값보다 낮은 경우
        if self.sma > self.data.close:
            # do something
            pass
        # 종가가 이동평균값보다 높은 경우
        elif self.sma < self.data.close:
            # do something else
            pass


def data_settings(code, year_start):
    price_df = fdr.DataReader(code, year_start)  # 비트코인 원화 가격 (빗썸) 2016년~현재
    # 결측치 존재 유무 확인
    invalid_data_cnt = len(price_df[price_df.isin([np.nan, np.inf, -np.inf]).any(1)])

    if invalid_data_cnt == 0:
        price_df['Date'] = price_df.index
        price_df['Adj Close'] = price_df['Close']
        df = price_df.loc[:, ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
        return df
    return False


class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.sma = bt.indicators.MovingAverageSimple(self.datas[0], period=15)  # 15일 이동평균 값
        self.rsi = bt.indicators.RelativeStrengthIndex()  # rsi 지수 정의

    def notify_order(self, order):
        if order.statusin[order.Submitted, order.Accepted]:
            return
        if order.statusin[order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price}, Cost: {order.executed.value}, Comm: {order.executed.comm}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(
                    f'SELLEXECUTED, Price: {order.executed.price}, Cose: {order.executed.value}, Comm: {order.executed.comm}')
            self.bar_executed = len(self)
        elif order.statusin[order.Canceled, order.Margin, order.Rejected]:
            self.log('OrderCanceled / Margin / Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return self.log(f'OPERATION PROFIT, GROSS {trade.pnl}, NET {trade.pnlcomm}')

    def next(self):
        if self.order:  # self.order 변수로 현재 미체결 내역 확인
            return
        if not self.position:  # 주문상태도 하니고 갖고 있는 포지션도 없다면 매수 조건에 맞는지 확인
            if self.rsi[0] < 30:  # rsi 지수가 30 이하면 매수
                self.log(f'BUY CREATE , {self.dataclose[0]}')
                self.order = self.buy(size=500)  # 500주 매수 주문
        else:
            if self.rsi[0] > 70:  # rsi 지수가 70 이상이면 매도
                self.log(f'SELL CREATE, {self.dataclose[0]}')
                self.order = self.sell(size=500)  # 500주 매도 주문


if __name__ == '__main__':
    # 전략을 활용할 자산은 미국 ETF 종목 중 하나인 QQQ (나스닥 100지수를 추종하는 ETF )다.
    # df = pd.read_csv('./QQQ.csv', index_col='DATE', parse_dates=['DATE'])
    df = data_settings('005930', '2018')
    # 우리가 갖고 있는 데이터를 Backtrader에서 사용할 수 있는 규격에 맞춰야 한다.
    # 사용할 데이터를 Backtrader에서 제공하는 PandasData 함수에 전달하고 추후 사용할 수 있는 형태로 반환한다.
    data = bt.feeds.PandasData(dataname=df)

    # backtrader에서 실질적으로 전략을 움직이는 Cerebro() 객체를 할당
    cerebro = bt.Cerebro()
    cerebro.addstrategy(TestStrategy)  # 전략을 추가
    cerebro.broker.setcommission(commission=0.001)  # 실제 거래에서 발생하게 될 수수료를 설정
    # cerebro.adddata(data, name=ticker)  # 변환된 데이터를 추가
    cerebro.adddata(data, name='005930')  # 변환된 데이터를 추가
    cerebro.broker.setcash(100000.0)  # 보유 현금 액수를 설정

    print(f'StartingPortfolioValue : {cerebro.broker.getvalue()}')
    cerebro.run()  # 전략을 수행
    print(f'FinalPortfolioValue: {cerebro.broker.getvalue()}')
    # 이미지를 저장하는 방식
    # (그림을 저장하려면 backtrader 라이브러리 내부 코드를 일부 수정해야 한다)
    cerebro.plot(volume=False, savefig=True, path='./backtrader-plot.png')
