챕터 별 페이지 내에서 ctrl + f 로 키워드 검색 가능하시면 빠른 검색 가능합니다. 

지속적으로 업데이트 될 예정이며, 모든 질문은 class101 게시판을 이용해 주시기 바랍니다. 

[TOC] 
  
# 데이터프레임 (dataframe)
데이터프레임 (dataframe)은 2차원 테이블 데이터 구조 (=행렬)를 가지는 자료형입니다.  

Dataframe은 각 열마다 다른 자료형 (dtype)을 가질 수 있습니다.  
**DataFrame**을 사용할 때에는 **D**와 **F**는 반드시 대문자를 사용해야 합니다.

# 데이터프레임 생성
## 기본형
Datafram 생성은 data에 입력되는 변수가 어떤 자료형을 가지는가에 따라 DataFrame() 함수 결과가 달라집니다. 

| 인자 (parameter) | 설명 | 
| -------- | -------- | 
| data     | 여러가지 자료형을 포함 가능 ([딕셔너리 (dictionary)](https://wikidocs.net/104376), 리스트의 리스트 (2차원), Series, 또 다른 DataFrame 등등)     | 
| index     | 행 (row)의 레이블 (label)의 리스트를 입력받는다.     | 
| columns     | 열 (column)의 레이블 (label)의 리스트를 입력받는다.     | 

`DataFrame = pandas.Dataframe(data, index=index, columns=colums`    

```{.python}
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
```


## 딕셔너리를 DataFrame 클래스의 생성자 인자로 전달 

DataFrame 객체를 생성하는 가장 쉬운 방법은 파이썬의 딕셔너리를 사용하는 것입니다. 딕셔너리를 통해 각 칼럼에 대한 데이터를 저장한 후 딕셔너리를 DataFrame 클래스의 생성자 인자로 넘겨주면 DataFrame 객체가 생성됩니다.


```{.python}
from pandas import DataFrame

company = {
    'code': ('BTC-KRW', 'ETH-KRW'),
    'name': ('비트코인', '이더리움'),
    'close': (20785000, 635800)
}
print(f"DataFrame 객체 생성 전 - type(company): {type(company)} ")           # company의 타입: 딕셔너리 (dict)
df_company = DataFrame(company)
print(f"DataFrame 객체 생성 후 - type(df_company): {type(df_company)} ")     # df_company의 타입: 데이터프레임 (DataFrame)
print(f"df_company의 열 (column) 길이: {len(df_company)}\n")
print(df_company)
# 출력
# DataFrame 객체 생성 전 - type(company): <class 'dict'>
# DataFrame 객체 생성 후 - type(df_company): <class 'pandas.core.frame.DataFrame'>
# df_company의 열 (column) 길이: 2
# 
#       code  name     close
# 0  BTC-KRW  비트코인  20785000
# 1  ETH-KRW  이더리움    635800
```

# 데이터 접근
## iloc - 행, 열 단위로 데이터에 접근

```{.python}
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
```
## loc - 행, 열 단위로 데이터에 접근 (loc - 열의 경우 라벨명으로 접근)

```{.python}
print(f"df_company.loc[0, 'code'] :{df_company.loc[0, 'code']}")        # 아래는 0행의 'code'열 출력
print(f"df_company.loc[0, 'name'] : {df_company.loc[0, 'name']}")       # 아래는 0행의 'name'열 출력
# 출력
# df_company.loc[0, 'code'] :BTC-KRW,
# df_company.loc[0, 'name'] : 비트코인
```


## 데이터프레임의 컬럼 순서 변경

```{.python}
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
```

