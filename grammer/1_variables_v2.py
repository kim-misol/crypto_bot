# 변수 값 숫자 (number: int, float)
a1 = 20210101
a2 = 10.5
print(a1)  # 20210101
print(type(a1))  # <class 'int'>
print(a2)  # 10.5
print(type(a2))  # <class 'float'>

# 변수 값 텍스트 (text: str)
b = "20210101"
print(b)  # 20210101
print(type(b))  # <class 'str'>

from datetime import datetime

# 변수 값 Boolean (참, 거짓: bool)
is_true = True
is_false = False
print(is_true)          # True
print(is_false)          # False
print(type(is_true))    # <class 'bool'>
print(type(is_false))    # <class 'bool'>

# 변수 값 날짜 (date)
c = datetime(2021, 1, 1)
print(c)  # 2021-01-01 00:00:00
print(type(c))  # <class 'datetime.datetime'>

# 변수 값 리스트 (list)
d = ['20210101', '20210102', '20210103']
print(d)  # ['20210101', '20210102', '20210103']
print(type(d))  # <class 'list'>
print(d[0])  # 20210101
print(d[1])  # 20210102
print(d[2])  # 20210103
print(d[-1])  # 20210103

# 변수 값 튜플 (tuple)
e = (0, '20210101')
print(e)  # (0, '20210101')
print(type(e))  # <class 'tuple'>
print(e[0])  # 0
print(e[1])  # 20210101

# F-Strings
year = 2021
month = 1
day = 1
day_of_week = "금요일"
# 쌍타옴표 앞에 f 입력
print(f"{year}년 {month}월 {day}일은 {day_of_week}입니다.")  # 2021년 1월 1일은 금요일입니다.
# 대문자 F도 가능
print(F"{year}년 {month}월 {day}일은 {day_of_week}입니다.")  # 2021년 1월 1일은 금요일입니다.

"""
[참고]

문자열 String Indexing slicing method docstring
https://velog.io/@ceres/Python-%EB%AC%B8%EC%9E%90%EC%97%B4-%EC%9D%B8%EB%8D%B1%EC%8B%B1-Indexing-1bk60g6n5y

List | Indexing, slicing, 수정, 삭제
https://velog.io/@ceres/Python-List
List | 메소드 list.append(), pop(), sort(), count()
https://velog.io/@ceres/Python-List-%EB%A9%94%EC%86%8C%EB%93%9C-list.append-sort-count

Tuple | Packing, Unpacking
https://velog.io/@ceres/Python-Tuple-Packing-Unpacking

for문 | 기본문,range(),중첩문
https://velog.io/@ceres/Python-for
for문 | comprehension
https://velog.io/@ceres/Python-for%EB%AC%B8-comprehension

연산자 | 할당, 산술, 문자열, 비교, 논리, 멤버쉽 연산자
https://velog.io/@ceres/Python-%EC%97%B0%EC%82%B0%EC%9E%90-%ED%95%A0%EB%8B%B9-%EC%82%B0%EC%88%A0-%EB%AC%B8%EC%9E%90%EC%97%B4-%EB%B9%84%EA%B5%90-%EB%85%BC%EB%A6%AC-%EB%A9%A4%EB%B2%84%EC%89%BD-%EC%97%B0%EC%82%B0%EC%9E%90
"""
