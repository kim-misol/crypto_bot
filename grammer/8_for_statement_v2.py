print("\n*************************************range()*************************************")
#
# 하나의 인자만 주어졌을 때, 0부터 인자값-1 까지의 수를 만들어준다
num_list = list(range(10))          # 0부터 9 (10-1)까지의 수
print(num_list)
# 출력값: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# range(start_value, end_value)
# 두 개의 인자를 주었을 때, 첫인자부터 두번째인자-1 까지의 만큼의 수를 만들어준다
num_list = list(range(2, 5))        # 2부터 4 (5-1)까지의 수
print(num_list)
# 출력값: [2, 3, 4]

# range(statr_value, end_value, step)
# 두 개의 인자를 주었을 때, 첫인자부터 두번째인자-1 까지의 만큼의 수를 만들어준다
num_list = list(range(1, 11, 2))    # 1부터 10까지의 수를 2의 간격으로 생성
print(num_list)
# 출력값: [1, 3, 5, 7, 9]


print("\n*************************************for문-range-0~9까지 출력*************************************")
num_list = []       # 빈 리스트 생성
print(num_list)
for i in range(10):
    num_list.append(i)
print(num_list)
# 출력
# []
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


print("\n*************************************for문-range-2~9까지 출력*************************************")
num_list = []       # 빈 리스트 생성
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