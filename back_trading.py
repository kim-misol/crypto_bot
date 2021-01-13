import datetime

import backtrader as bt
import pandas_datareader as web
import numpy as np


class TestStrategy(bt.Strategy):

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.i = 0
        self.pastMA5 = 0
        self.pastMA20 = 0
        self.meanPrice = 0
        self.numberOfStocks = 0

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        elif order.status in [order.Completed]:
            if order.isbuy():
                self.meanPrice = (self.meanPrice * self.numberOfStocks + order.size * self.dataclose[0]) / (
                            self.numberOfStocks + order.size)
                self.numberOfStocks += order.size
            elif order.issell():
                self.meanPrice = 0
                self.numberOfStocks = 0
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print('Order Canceled/Margin/Rejected')
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        # print(str(self.datas[0].datetime.date(0)) + ' PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))

    def next(self):
        self.i += 1
        if self.i < 20:
            return
        else:
            MA5 = np.mean([self.dataclose[n - 4] for n in range(5)])
            MA20 = np.mean([self.dataclose[n - 19] for n in range(20)])
            if MA5 > MA20 and self.pastMA5 < self.pastMA20 and self.broker.getcash() > self.dataclose[0]:
                amountToOrder = int(self.broker.getcash() / self.dataclose[0])
                self.order = self.buy(size=amountToOrder)
                print(self.i, " - BUY : {}, buy amount : {}".format(self.dataclose[0], amountToOrder))

            elif self.meanPrice * 1.1 < self.dataclose[0] and self.numberOfStocks != 0:
                print(self.i, " - SELL : {}, sell amount : {}".format(self.dataclose[0], self.numberOfStocks))
                self.order = self.sell(size=self.numberOfStocks)

            self.pastMA5 = MA5
            self.pastMA20 = MA20

# 1
stockData = web.DataReader('AAPL', 'yahoo', datetime.datetime(2016, 6, 1), datetime.datetime(2018, 2, 1))[
    ['Open', 'Close']]

# 2
cerebro = bt.Cerebro()
cerebro.broker.setcash(100000.0)
cerebro.broker.setcommission(commission=0.001)

# 3
cerebro.addstrategy(TestStrategy)

# 4
dts = bt.feeds.PandasData(dataname=stockData, open='Open', close='Close')
cerebro.adddata(dts)

# 5
print('Starting portfolio Value : {}'.format(cerebro.broker.getvalue()))
cerebro.run()
print('Final portfolio value : {}'.format(cerebro.broker.getvalue()))

# cerebro.plot()
