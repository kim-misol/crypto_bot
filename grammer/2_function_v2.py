print("\n*************************************함수 내에서 결과값 출력*************************************")


def sum(a, b):
    res = a + b
    print(res)


print("함수 호출 전")
sum(1, 2)
print("함수 호출 후")

print("\n*************************************함수 내에서 결과값 리턴*************************************")


def sum(a, b):
    return a + b


print("함수 호출 전")
result = sum(1, 2)
print("함수 호출 후")
print(result)

print("\n*************************************기본 형태*************************************")


## 매개변수(Paramete), 인자 (Argument)

# Parameter: 매개변수 - 함수와 메서드 입력 변수(Variable) 명
# Argument: 인자 - 함수와 메서드의 입력 값(Value)
def sum(a, b):  # a, b: Parameter 함수정의 부분에 나열
    return a + b


result = sum(1, 2)  # 1, 2: argument 함수 호출시 전달되는 실제 값
print(result)

print("\n*************************************keyword argument*************************************")


# 예시 2 - Keyword Argument: parameter에 맞추어 값을 전해준다. (paramter 순서가 변경되어도 된다)
def print_date(month, day):
    print(f"{month}월 {day}일")


print_date(month=12, day=31)  # 12월 31일

print("\n*************************************parameter default value*************************************")


# 예시 3 - Parameter default 값 정의 (defaut 값이 정의된 parameter는 함수 호출 시 인자를 넘겨주지 않아도 된다)
def print_date(month=1, day=1):
    print(f"{month}월 {day}일")


print_date(month=12, day=31)  # 12월 31일
print_date()  # 1월 1일

print("\n*********************************mixing keyword argument and default value*********************************")


# 예시 4 - Parameter default 값 정의
# default 값이 정의 되지 않은 parameter 부터 작성 후 default 값이 정의된 parameter가 와야한다.
def print_date(month, day=1):
    print(f"{month}월 {day}일")


print_date(month=12, day=31)  # 12월 31일
# print_date()  # TypeError: print_date() missing 1 required positional argument: 'month'

def print_date(month, day=1):
    print(f"{month}월 {day}일")


print_date(month=12)  # 12월 1일

# print_date()  # TypeError: print_date() missing 1 required positional argument: 'month'

# def print_date(day=1, month):   # SyntaxError: non-default argument follows default argument
#     print(f"{month}월 {day}일")
#
#
# print_date(month=12, day=31)
