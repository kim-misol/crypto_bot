print("\n*************************************range()*************************************")
#
# 하나의 인자만 주어졌을 때, 0부터 인자값-1 까지의 수를 만들어준다
num_list = list(range(10))  # 0부터 9 (10-1)까지의 수
print(num_list)
# 출력값: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# range(start_value, end_value)
# 두 개의 인자를 주었을 때, 첫인자부터 두번째인자-1 까지의 만큼의 수를 만들어준다
num_list = list(range(2, 5))  # 2부터 4 (5-1)까지의 수
print(num_list)
# 출력값: [2, 3, 4]

# range(statr_value, end_value, step)
# 두 개의 인자를 주었을 때, 첫인자부터 두번째인자-1 까지의 만큼의 수를 만들어준다
num_list = list(range(1, 11, 2))  # 1부터 10까지의 수를 2의 간격으로 생성
print(num_list)
# 출력값: [1, 3, 5, 7, 9]


print("\n*************************************for문-range-0~9까지 출력*************************************")
num_list = []  # 빈 리스트 생성
print(num_list)
for i in range(10):
    num_list.append(i)
print(num_list)
# 출력
# []
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


print("\n*************************************for문-range-2~9까지 출력*************************************")
num_list = []  # 빈 리스트 생성
print(num_list)
for j in range(2, 10):
    num_list.append(j)
print(num_list)
# 출력
# []
# [2, 3, 4, 5, 6, 7, 8, 9]


print("\n*************************************2중for문*************************************")
'''
for 변수 1 in Sequence 1(list, tuple, range ....) :
    수행할 문장 1
    for 변수 2 in Sequence 2(list, tuple, range....):
        수행할 문장 2
'''
date_list = [('2020', '1', '1'), ('2020', '1', '2'), ('2020', '1', '3')]
for i in range(len(date_list)):  # date_list의 길이만큼 for문 돌기
    date = ''  # 새로운 date을 출력하기 위해서 reset
    for j in range(len(date_list[i])):  # date_list 리스트 안의 tuple의 길이만큼 for문 돌기
        date += date_list[i][j] + ' '
    print(date)
# 출력
# 2020 1 1
# 2020 1 2
# 2020 1 3

print("\n*************************************break*************************************")
num_list = [3, 4, 2, 0, 5, 9]

for i in num_list:
    if i == 0:
        break
    print(f"12 나누기 {i} = {12 / i}")
# 출력
# 12 나누기 3 = 4.0
# 12 나누기 4 = 3.0
# 12 나누기 2 = 6.0


print("\n*************************************continue*************************************")
for i in range(1, 5):
    if i == 3:
        continue  # i가 3인 경우 현재 루프를 건너뛰므로 i가 3일 때는 출력하지 않는다.
    print(i)
# 출력값: 1, 2, 4

date_list = [('2020', '1', '1'), ('2020', '1', '2'), ('2020', '1', '3')]
for i in range(len(date_list)):
    date = ''
    for j in range(len(date_list[i])):
        if j == len(date_list[i]) - 1:  # j가 튜플 마지막 인덱스일 때 (j == 2), 값을 date에 더해준 후 아래의 명령어를 실행하지 않고 다음 루프로 넘어간다.
            date += date_list[i][j]
            continue
        date += date_list[i][j] + '-'  # continue에 의해서 j == len(date_list[i]) - 1 조건을 만족하지 않았을 경우에만 실행된다.
    print(date)
# 출력
# 2020-1-1
# 2020-1-2
# 2020-1-3

print("\n*************************************pass*************************************")
for i in range(1, 5):
    if i == 3:
        pass  # i가 3인 경우 pass -> 코드 실행에 아무런 영향을 주지 않는다.
    print(i)
# 출력값: 1, 2, 3, 4

print("\n*************************************else*************************************")
num_list = [3, 4, 2, 0, 5, 9]

for i in num_list:
    if i == 0:
        continue
    print(f"12 나누기 {i} = {12 / i}")
else:
    print("계산 완료\n")
# 출력
# 12 나누기 3 = 4.0
# 12 나누기 4 = 3.0
# 12 나누기 2 = 6.0
# 12 나누기 5 = 2.4
# 12 나누기 9 = 1.3333333333333333
# 계산 완료

for i in num_list:
    if i == 0:
        break
    print(f"12 나누기 {i} = {12 / i}")
else:
    print("계산 완료")
# 출력
# 12 나누기 3 = 4.0
# 12 나누기 4 = 3.0
# 12 나누기 2 = 6.0
