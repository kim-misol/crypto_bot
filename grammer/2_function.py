# 함수란
# def는 define의 앞글자를 따서 만든거다.
# 함수명은 자유롭게 기능에 맞춰 지어내면 된다.
# 함수 내에서는 tab 키로 한 칸 띄우고 입력해야한다.
# 함수 이름은 중복되면 안된다.
# 함수는 호출하기 이전에 정의해주어야 한다.

print("\n*************************************ex1*************************************")


def print_test():
    print("test!!!!")


# 함수를 호출할 때는 함수 이름과 "()" 를 붙여서 사용하면 된다.
# 함수는 호출되기 전까지는 구동 하지 않는다.
print("함수 호출 전")
print_test()
print("함수 호출 후")

print("\n*************************************ex2*************************************")


# 매개변수(parameter)
def print_code_name(name):
    print(name)


print_code_name("비트코인")
print_code_name("이더리움")

print("\n*************************************ex3*************************************")
# 함수를 사용하는 이유? 코드가 중복되는 것을 방지하기 위해서
# ex) 각각의 평균 값을 구하여 출력하는 프로그램을 만드시오.
high = 20460000
low = 19937000
average = (high + low) / 2
print(average)

high_d1 = 20538000
low_d1 = 20307000
average_d1 = (high_d1 + low_d1) / 2
print(average_d1)

high_d2 = 20595000
low_d2 = 20348000
average_d2 = (high_d2 + low_d2) / 2
print(average_d2)


print("\n*************************************ex4*************************************")


# 위 코드를 함수로 간소화시키면???
def print_average(high, low):
    avg = (high + low) / 2
    print(avg)


print_average(20460000, 19937000)
print_average(20538000, 20307000)
print_average(20595000, 20348000)

print("\n*************************************ex5*************************************")


# return
def return_average(high, low):
    return (high + low) / 2


average = return_average(20460000, 19937000)

print(average)

print("\n*************************************ex6*************************************")


# 지역변수 vs 전역 변수
# 함수 내에서 선언된 변수를 지역변수, 함수 밖에서 선언된 변수는 전역변수
# 지역 변수는 함수가 호출되었을 때에만 저장되어 있다. 즉, 함수를 빠져나오는 순간 저장된 값은 사라진다.

def yesterday():
    day_of_week = "Monday"
    print("지역변수 : ", day_of_week)


day_of_week = "Tuesday"
yesterday()
print("전역변수 :", day_of_week)

print("\n*************************************ex6_1*************************************")


def yesterday_v2():
    day_of_week = "Sunday"
    print(day_of_week)


day_of_week = "Monday"
yesterday_v2()
print(day_of_week)
