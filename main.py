import FinanceDataReader as fdr
import numpy as np
import matplotlib.pylab as plt

from trading_indicators import *


def create_trade_book(sample):
    book = sample[['Close']].copy()
    book['trade'] = ''
    return book


def tradings(sample, book):
    for i in sample.index:
        if sample.loc[i, 'Close'] > sample.loc[i, 'ubb']:  # 상단 밴드 이탈 시 동작 안함
            book.loc[i, 'trade'] = ''
        elif sample.loc[i, 'lbb'] > sample.loc[i, 'Close']:  # 하반 밴드 이탈 시 매수
            if book.shift(1).loc[i, 'trade'] == 'buy':  # 이미 매수 상태라면
                book.loc[i, 'trade'] = 'buy'  # 매수상태 유지
            else:
                book.loc[i, 'trade'] = 'buy'
        elif sample.loc[i, 'lbb'] <= sample.loc[i, 'Close'] <= sample.loc[i, 'ubb']:  # 볼린저 밴드 안에 있을 시
            if book.shift(1).loc[i, 'trade'] == 'buy':
                book.loc[i, 'trade'] = 'buy'  # 매수상태 유지
            else:
                book.loc[i, 'trade'] = ''  # 동작 안 함
    return book


def returns(book):
    # 손익 계산
    rtn = 1.0
    book['return'] = 1
    buy = 0.0
    sell = 0.0

    for i in book.index:
        # long 매수
        if book.loc[i, 'trade'] == 'buy' and book.shift(1).loc[i, 'trade'] == '':
            buy = book.loc[i, 'Close']
            print('매수 날짜 : ', i, ' 매수 가격 : ', buy)
        # long 매도
        elif book.loc[i, 'trade'] == '' and book.shift(1).loc[i, 'trade'] == 'buy':
            sell = book.loc[i, 'Close']
            rtn = (sell - buy) / buy + 1  # 손익 계산
            book.loc[i, 'return'] = rtn
            print('매도 날짜 : ', i, ' 매수 가격 : ', buy, ' | 매도 가격 : ', sell, ' | return :', round(rtn, 4))

        if book.loc[i, 'trade'] == '':  # 제로 포지션
            buy = 0.0
            sell = 0.0

    # 누적 수익률 계산
    acc_rtn = 1.0
    for i in book.index:
        rtn = book.loc[i, 'return']
        acc_rtn = acc_rtn * rtn  # 누적 수익률 계산
        book.loc[i, 'acc return'] = acc_rtn
    print('Accunulated return :', round(acc_rtn, 4))

    return round(acc_rtn, 4)


if __name__ == "__main__":
    price_df = fdr.DataReader('BTC/KRW', '2018')  # 비트코인 원화 가격 (빗썸) 2016년~현재
    # 결측치 존재 유무 확인
    invalid_data_cnt = len(price_df[price_df.isin([np.nan, np.inf, -np.inf]).any(1)])

    if invalid_data_cnt == 0:
        df = price_df.loc[:, ['Close']].copy()
        bb_df = bollinger_band(df)

        base_date = '2018-01-01'
        sample = bb_df.loc[base_date:]
        # 볼린더밴드 지표 수치 추가
        book = create_trade_book(sample)
        book = tradings(sample, book)
        earn = returns(book)
        print(book.tail(100))

        # 변화 추이
        book['acc return'].plot()
        plt.show()
