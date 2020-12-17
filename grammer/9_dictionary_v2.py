print("\n************************************* dictionary 선언 *************************************")
company = {}
print(f"company: {company}")
print(f"type: {type(company)}")
# 출력
# test: {}
# type: <class 'dict'>

# company = dict(비트코인='BTC-KRW', 이더리움='ETH-KRW')
# print(f"company: {company}")
# # 출력
# # company: {'비트코인': 'BTC-KRW', '이더리움': 'ETH-KRW'}

print("\n************************************* 특정 key값의 value 출력 *************************************")
company = {'비트코인': 'BTC-KRW', '이더리움': 'ETH-KRW'}

print(f"company: {company}, type: {type(company)}")
print("비트코인 : ", company['비트코인'])         # 비트코인 key 값을 가진 value를 출력
# 출력
# company: {'비트코인': 'BTC-KRW', '이더리움': 'ETH-KRW'} , type: <class 'dict'>
# 비트코인 :  BTC-KRW

print("\n************************************* key, value 추가 *************************************")
company = {'비트코인': 'BTC-KRW', '이더리움': 'ETH-KRW'}
print(f"company 리플 추가 전: {company}")
company['리플'] = 'XRP-KRW'                       # key값 '리플'을 가지는 value 'XRP-KRW'를 company 딕셔너리에 추가
print(f"company 리플 추가 후: {company}")
# 출력
# company 리플 추가 전: {'비트코인': 'BTC-KRW', '이더리움': 'ETH-KRW'}
# company 리플 추가 후: {'비트코인': 'BTC-KRW', '이더리움': 'ETH-KRW', '리플': 'XRP-KRW'}

print("\n************************************* dictionary 수정 *************************************")
company = {'비트코인': 'BTC-KRW', '이더리움': 'BTC-KRW'}
print(f"company 수정 전: {company}")
company['이더리움'] = 'ETH-KRW'                 # 이더리움 key 값을 가진 value를 'ETH-KRW'으로 대체
print(f"company 수정 후: {company}")
# 출력
# company 수정 전: {'비트코인': 'BTC-KRW', '이더리움': 'BTC-KRW'}
# company 수정 후: {'비트코인': 'BTC-KRW', '이더리움': 'ETH-KRW'}

print("\n************************************* for문 활용 *************************************")
company = {'비트코인': 'BTC-KRW', '이더리움': 'ETH-KRW', '리플': 'XRP-KRW'}
for key in company:
    val = company[key]
    print(f"key : {key}, val : {val}")
# 출력
# key : 비트코인, val : BTC-KRW
# key : 이더리움, val : ETH-KRW
# key : 리플, val : XRP-KRW

print("\n************************************* key, value 값 가져오기 *************************************")
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
print(type(keys))
print(type(values))
for v in values:
    print(v)
# 출력
# BTC-KRW
# ETH-KRW
# XRP-KRW

print("\n************************************* items() 함수 *************************************")
company = {'비트코인': 'BTC-KRW', '이더리움': 'ETH-KRW', '리플': 'XRP-KRW'}
# items 값 가져오기
items = company.items()
print(type(items))

for item in items:
    print(f"item: {item}, key: {item[0]}, value: {item[1]}")
    print(type(item))

# 출력
# item: ('비트코인', 'BTC-KRW'), key: 비트코인, value: BTC-KRW
# item: ('이더리움', 'ETH-KRW'), key: 이더리움, value: ETH-KRW
# item: ('리플', 'XRP-KRW'), key: 리플, value: XRP-KRW


