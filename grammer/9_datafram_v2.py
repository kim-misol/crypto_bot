print("\n************************************* 데이터프레임 생성 *************************************")
# DataFrame ? => 행렬을 저장
# pandas라는 모듈의 DataFrame 클래스를 import
from pandas import DataFrame

data = [['BTC-KRW', '비트코인', 20785000], ['ETH-KRW', '이더리움', 635800]]
len_data = 2
df = DataFrame(data, index=range(2), columns=['code', 'name', 'close'])
print(df)
# 출력
#       code  name     close
# 0  BTC-KRW  비트코인  20785000
# 1  ETH-KRW  이더리움    635800

print("\n************************************* 딕셔너리를 데이터프레임으로 변환 *************************************")
company = {
    'code': ('BTC-KRW', 'ETH-KRW'),
    'name': ('비트코인', '이더리움'),
    'close': (20785000, 635800)
}
print(f"변환 전 타입: {type(company)} ")           # company의 타입: 딕셔너리 (dict)
df_company = DataFrame(company)
print(f"변환 후 타입: {type(df_company)} ")     # df_company의 타입: 데이터프레임 (DataFrame)
print(f"df_company의 열 (column) 길이: {len(df_company)}\n")
print(df_company)
# 출력
# 변환 전 타입: <class 'dict'>
# 변환 후 타입: <class 'pandas.core.frame.DataFrame'>
# df_company의 열 (column) 길이: 2
#
#       code  name     close
# 0  BTC-KRW  비트코인  20785000
# 1  ETH-KRW  이더리움    635800

print("\n************************************* iloc - 행, 열 단위로 데이터에 접근 *************************************")
'''
iloc : 행, 열 단위로 DataFrame 안에 있는 데이터에 접근
'''
company = {
    'code': ('BTC-KRW', 'ETH-KRW'),
    'name': ('비트코인', '이더리움'),
    'close': (20785000, 635800)
}
df_company = DataFrame(company)
print(f"df_company.iloc[0, 0] :{df_company.iloc[0, 0]}")
print(f"df_company.iloc[0, 1] : {df_company.iloc[0, 1]}")
print(f"df_company.iloc[0, 2] : {df_company.iloc[0, 2]}")
# 출력
# df_company.iloc[0, 0] :BTC-KRW
# df_company.iloc[0, 1] : 비트코인
# df_company.iloc[0, 2] : 20785000

print("\n************************************* loc - 열의 경우 라벨명으로 접근 *************************************")
'''
loc:  iloc와 동일하게 DataFrame에 행, 열로 접근하는데 열의 경우 라벨명으로 접근.
'''

print(f"df_company.loc[0, 'code'] :{df_company.loc[0, 'code']}")        # 아래는 0행의 'code'열 출력
print(f"df_company.loc[0, 'name'] : {df_company.loc[0, 'name']}")       # 아래는 0행의 'name'열 출력
# 출력
# df_company.loc[0, 'code'] :BTC-KRW,
# df_company.loc[0, 'name'] : 비트코인

print("\n************************************* 칼럼 순서 변경 가능 *************************************")
company = {
    'code': ('BTC-KRW', 'ETH-KRW', 'XRP-KRW'),
    'name': ('비트코인', '이더리움', '리플'),
    'close': (20785000, 635800, 552)
}

print(f"칼럼 순서 변경 전 :\n {df_company}\n")
df_company = DataFrame(company, columns=['close', 'code', 'name'])
print(f"칼럼 순서 변경 후 :\n {df_company}")
# 출력
# 칼럼 순서 변경 전 :
#        code  name     close
# 0  BTC-KRW  비트코인  20785000
# 1  ETH-KRW  이더리움    635800
#
# 칼럼 순서 변경 후 :
#        close     code  name
# 0  20785000  BTC-KRW  비트코인
# 1    635800  ETH-KRW  이더리움
# 2       552  XRP-KRW    리플