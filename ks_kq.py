# https://chancoding.tistory.com/108
from pandas_datareader import data

# 두 가지의 방식
# 방법 1
# df = data.DataReader("^KS11", "yahoo")
# 방법 2
df_kospi = data.get_data_yahoo("^KS11")
df_kosdaq = data.get_data_yahoo("^KQ11")
df_nasdaq = data.get_data_yahoo('^IXIC')

# 날짜 지정
# start_date = datetime(2007,1,1)
# end_date = datetime(2020,3,3)
#
# df = data.get_data_yahoo("^KS11", start_date, end_date)
