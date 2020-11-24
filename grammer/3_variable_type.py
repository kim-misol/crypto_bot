"""
[자료형]
자료형(data type) : 데이터의 종류
 * 자료형 종류
    정수: int
    실수: float
    문자열: str
    (참, 거짓): bool

    list(리스트)
    tuple(튜플)
    set(집합, 셋)
    dict(사전, 딕셔너리)
    등등
"""

print("\n*************************************ex1*************************************")
# type() : 데이터 타입을 확인 하는 내장 함수
print("1: ", type(1))
print("1.1: ", type(1.1))
print("abc: ", type("abc"))
print("True: ", type(True))
print("False: ", type(False))

code_name = "비트코인"
current_price = 20300000
print("\ntype code_name: ", type(code_name))
print("type current_price: ", type(current_price))


print("\n*************************************ex2*************************************")
'''문자열 합치기'''

db_name = 'CoinBot'
imi_num = "_imi1"
db_name_imi = db_name + imi_num
print(db_name_imi)

print("\n*************************************ex3*************************************")
'''문자열 포맷팅'''

code_name = '비트코인'
current_price = 20300000
print("%s 현재가: %s" % (code_name, current_price))  # 비추천, python2 버전에 사용하던 방식
print("{} 현재가: {}".format(code_name, current_price))  # 추천
print(f"{code_name} 현재가: {current_price}")  # 추천 / f-string 이라고 표현 / python 3.6 이상 부터 지원
# 비교
print(code_name, "현재가:", current_price)
