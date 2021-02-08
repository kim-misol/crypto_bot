챕터 별 페이지 내에서 ctrl + f 로 키워드 검색 가능하시면 빠른 검색 가능합니다. 

지속적으로 업데이트 될 예정이며, 모든 질문은 class101 게시판을 이용해 주시기 바랍니다. 

[TOC] 
  
# 딕셔너리 (dictionary)
딕셔너리는 불변한 키(key)와 가변한 값(value)으로 맵핑되어 있는 순서가 없는 집합입니다.


| 키(key) | 값(value) |
| -------- | -------- |
| 불변     | 가변     | 



### 딕셔너리 (dictionary) 선언 
딕셔너리 선언할 때는 빈 중괄호 `{}`를 사용합니다.  
```{.python}
company = {}
print(f"company: {company}")
print(f"type: {type(company)}")                     # company의 변수 타입 출력
# 출력
# test: {}
# type: <class 'dict'>
```


### 특정 key값의 value 출력 
특정 `key` 값의 `value`를 출력할 때는 딕셔너리 이름 옆에 대괄호 `[]` 를 붙이고 대괄호 안에 `key` 값을 넣어 `value` 값을 가져옵니다.
```{.python}
company = {'비트코인': 'BTC-KRW', '이더리움': 'ETH-KRW'}

print(f"company: {company}, type: {type(company)}")
print("비트코인 : ", company['비트코인'])         # 비트코인 key 값을 가진 value를 출력
# 출력
# company: {'비트코인': 'BTC-KRW', '이더리움': 'ETH-KRW'} , type: <class 'dict'>
# 비트코인 :  BTC-KRW
```

### key, value 추가  

```{.python}
company = {'비트코인': 'BTC-KRW', '이더리움': 'ETH-KRW'}
print(f"company 리플 추가 전: {company}")
company['리플'] = 'XRP-KRW'                       # key값 '리플'을 가지는 value 'XRP-KRW'를 company 딕셔너리에 추가
print(f"company 리플 추가 후: {company}")
# 출력
# company 리플 추가 전: {'비트코인': 'BTC-KRW', '이더리움': 'ETH-KRW'}
# company 리플 추가 후: {'비트코인': 'BTC-KRW', '이더리움': 'ETH-KRW', '리플': 'XRP-KRW'}
```

### 특정 key값의 value 수정  
특정 `key`값의 `value`를 출력할 때와 같이 딕셔너리 이름 옆에 대괄호 `[]` 를 붙이고 대괄호 안에 `key` 값을 넣은 후 다른 value로 대체해줍니다.
```{.python}
company = {'비트코인': 'BTC-KRW', '이더리움': 'BTC-KRW'}
print(f"company 수정 전: {company}")
company['이더리움'] = 'ETH-KRW'                 # 이더리움 key 값을 가진 value를 'ETH-KRW'으로 대체
print(f"company 수정 후: {company}")
# 출력
# company 수정 전: {'비트코인': 'BTC-KRW', '이더리움': 'BTC-KRW'}
# company 수정 후: {'비트코인': 'BTC-KRW', '이더리움': 'ETH-KRW'}
```
  
### for문 활용  

```{.python}
company = {'비트코인': 'BTC-KRW', '이더리움': 'ETH-KRW', '리플': 'XRP-KRW'}
for key in company:
    val = company[key]
    print(f"key : {key}, val : {val}")
# 출력
# key : 비트코인, val : BTC-KRW
# key : 이더리움, val : ETH-KRW
# key : 리플, val : XRP-KRW
```

### keys, values 값 가져오기  

`keys()`와 `values()` 함수를 이용하여 딕셔너리 키들과 값들을 가져올 수 있습니다.

```{.python}
company = {'비트코인': 'BTC-KRW', '이더리움': 'ETH-KRW', '리플': 'XRP-KRW'}
# keys 값 가져오기
keys = company.keys()
for k in keys:
    print(k)
# 출력
# 비트코인
# 이더리움
# 리플

# values 값 가져오기
values = company.values()
for v in values:
    print(v)
# 출력
# BTC-KRW
# ETH-KRW
# XRP-KRW
```

### items 값 가져오기  

`items()` 함수를 이용하여 딕셔너리 항목들((key, value) 쌍들)을 가져올 수 있으며,  
딕셔너리 항목은 (key, value) 쌍이 [tuple](https://wikidocs.net/104374) 형태로 있습니다.

```{.python}
company = {'비트코인': 'BTC-KRW', '이더리움': 'ETH-KRW', '리플': 'XRP-KRW'}
# items 값 가져오기
items = company.items()
for item in items:
    print(f"item: {item}, key: {item[0]}, value: {item[1]}")

# 출력
# item: ('비트코인', 'BTC-KRW'), key: 비트코인, value: BTC-KRW
# item: ('이더리움', 'ETH-KRW'), key: 이더리움, value: ETH-KRW
# item: ('리플', 'XRP-KRW'), key: 리플, value: XRP-KRW
```