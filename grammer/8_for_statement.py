print("\n*************************************ex0*************************************")

# 0~9까지 출력하는 프로그램을 작성하라.
print(0)
print(1)
print(2)
print(3)
print(4)
print(5)
print(6)
print(7)
print(8)
print(9)

print("\n*************************************ex1*************************************")

'''
range 함수 : range는 연속된 숫자를 만들어준다
'''
# 하나의 인자만 주어졌을 때, 0 ~ 인자값-1 만큼의 수를 만들어준다
list1 = list(range(10))
print(list1)

# 두 개의 인자를 주었을 때, 첫인자 ~ 두번째인자-1 만큼의 수를 만들어준다

list2 = list(range(2, 5))
print(list2)

'''
Sequence : 반복 가능한 형식, str(str은 여러개의 문자를 가지고 있는 Sequence), list, dict 등의 자료형

for 문이란 ? 반복문이다.
* 기본 구조 
    for 변수 in Sequence(list, 튜플, range ....) : 
        수행 문장 

for 문은 Sequence 원소를 차례대로 한 개씩 받아 온다. 모든 원소를 받아 올 때 까지 반복된다.
'''
# for문을 사용하여 0~9까지 출력하는 프로그램을 작성하라.
# range : 연속된 숫자를 만든다.
for i in range(10):
    print(i)

# 2~9까지 출력하는 프로그램을 작성하라.
print("\n*************************************ex2*************************************")
for k in range(2, 10):
    print(k)

print("\n*************************************ex3*************************************")
# 아래 리스트 a에 담긴 모든 원소(요소,element)를 출력하는 프로그램을 만드시오.
a = [1, 2, 3, 4, 5]
print(a[0])
print(a[1])
print(a[2])
print(a[3])
print(a[4])

print("\n*************************************ex4*************************************")
# 위 프로그램을 for문으로 수정하여 같은 결과를 출력하는 프로그램을 만드시오.
for item in a:
    print(item)

print("\n*************************************ex5*************************************")
b = ["안녕", "하세요", "저는", "홍길동", "입니다"]
for i in b:
    print(i)

print("\n*************************************ex6*************************************")
# 2중 for 문
for first in [('a', 'b', 'c'), ('d', 'e', 'f'), ('e', 'f', 'g')]:
    print("first: ", first)
    for second in first:
        print("second: ", second)

print("\n*************************************ex7*************************************")


# 예를들어 우리가 OpenAPI를 통해 모 회사의 1월 1일부터 1월 8일까지의 일봉데이터를 daily_coin_price에 저장했다고 가정하자


def save_to_db(data):
    # DB에 넣는 함수를 만들었다고 가정
    print(f"{data}를 DB에 넣습니다...")


daily_coin_price = [
    ('2020-01-01', '8300000', '비트코인'), ('2020-01-02', '8037000', '비트코인'), ('2020-01-03', '8474000', '비트코인'),
    ('2020-01-04', '8482000', '비트코인'), ('2020-01-05', '8454000', '비트코인'), ('2020-01-06', '8866000', '비트코인'),
    ('2020-01-07', '9398000', '비트코인'), ('2020-01-08', '9085000', '비트코인')
]

# 이 데이터를 우리 DB에 차곡 차곡 넣고 싶을 때 직접 한 개 한 개 index를 지정해서 가져오면 힘들 것이다.
save_to_db(daily_coin_price[0])
save_to_db(daily_coin_price[1])
save_to_db(daily_coin_price[2])
save_to_db(daily_coin_price[3])
save_to_db(daily_coin_price[4])
save_to_db(daily_coin_price[5])
save_to_db(daily_coin_price[6])
save_to_db(daily_coin_price[7])

# 만약 1년 치를 가져왔다고 생각해보면 코드는 끔찍하게 길어질 것이다. 이럴 때 아래와 같이 for문을 사용하면 간편해진다.
for item in daily_coin_price:
    save_to_db(item)


# 미션 - daily_coin_price 리스트에 담겨있는 삼성전자의 데이터를 이용하여 5일동안의 평균값을 계산하여
daily_coin_price = [
    ('2020-01-01', 8300000, '비트코인'), ('2020-01-02', 8037000, '비트코인'), ('2020-01-03', 8474000, '비트코인'),
    ('2020-01-04', 8482000, '비트코인'), ('2020-01-05', 8454000, '비트코인')
]

sum_price = 0
for item in daily_coin_price:
    sum_price += item[1]
clo5 = sum_price / 5
print(clo5)

# 미션
# 영상을 보고 date_rows와 time_rows 리스트에 담겨있는 날짜와 시간 데이터를 이용하여
# 이중 for문과 문자열 합산을 통해 아래 예시 출력값과 같이
# 출력한 뒤 Console 창을 캡쳐하여 올려주세요!
['202011230900', '202011230930', '202011231000', '202011231030', '202011240900', '202011240930', '202011241000',
'202011241030', '202011250900', '202011250930', '202011251000', '202011251030', '202011260900', '202011260930',
'202011261000', '202011261030', '202011270900', '202011270930', '202011271000', '202011271030']
date_rows = ["20201123", "20201124", "20201125", "20201126", "20201127"]
time_rows = ["0900", "0930", "1000", "1030"]
min_rows = []
for data in date_rows:
    for time in time_rows:
        min_rows.append(data+time)
print(min_rows)

min_rows = []
for i in range(10):
    min_rows.append(i)
print(min_rows)