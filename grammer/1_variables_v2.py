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

print("\n************************************* String formatting *************************************")
# F-Strings
year = 2021
month = 1
day = 1
day_of_week = "금요일"
# 쌍타옴표 앞에 f 입력
print(f"{year}년 {month}월 {day}일은 {day_of_week}입니다.")  # 2021년 1월 1일은 금요일입니다.
# 대문자 F도 가능
print(F"{year}년 {month}월 {day}일은 {day_of_week}입니다.")  # 2021년 1월 1일은 금요일입니다.

print("\n************************************* multi line *************************************")
# 작은따옴표 3개(''') 또는 큰따옴표 3개(""")를 사용하면 여러줄인 문자열을 변수에 대입할 수 있습니다.
multi_line = """가나다
라마바
사아자"""

print(f"{multi_line}")
# 출력
# 가나다
# 라마바
# 사아자

print("\n************************************* '{}'.format() *************************************")
a = "{0}년 {1}월 {2}일".format(2021, 1, 1)
print(a)
# 출력: 2021년 1월 1일
b = "{2}년 {1}월 {0}일".format(1, 1, 2021)
print(b)
# 출력: 2021년 1월 1일

c = "{0}년 {1}월 {2}일 {3}".format(2021, 1, 1, '금요일')
print(c)
# 출력: 2021년 1월 1일 금요일

year = 2021
month = 1
day = 1
day_of_week = '금요일'
d = "{0}년 {1}월 {2}일 {3}".format(year, month, day, day_of_week)
print(d)
# 출력: 2021년 1월 1일 금요일

e = "{year}년 {month}월 {day}일 {day_of_week}".format(year=2021, month=1, day=1, day_of_week='금요일')
print(e)
# 출력: 2021년 1월 1일 금요일
