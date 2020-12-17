print("\n************************************* list index *************************************")
a = ['p', 'r', 'o', 'b', 'e']
print(a[0])     # p
print(a[-5])    # p

print(a[1])     # r
print(a[-4])    # r

print(a[2])     # o
print(a[-3])    # o

print(a[3])     # b
print(a[-2])    # b

print(a[4])     # e
print(a[-1])    # e

print("\n*************************************indexing-원하는 value 가져오기*************************************")

alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
             'v', 'w', 'x', 'y', 'z']

# len()은 alphabets 라는 리스트의 길이를 알려주는 내장 함수
print("len(alphabets): ", len(alphabets))

# alphabets 의 0번째 인덱스의 원소(값)
print("alphabets[0]: ", alphabets[0])
# alphabets 의 4번째 인덱스의 원소(값)
print("alphabets[4]:", alphabets[4])

# alphabets[-1] 처럼 인덱스가 -1인 경우는 리스트의 뒤에서 첫번째 (마지막) 원소가 출력
print("alphabets[-1]:", alphabets[-1])

# alphabets[-2] 처럼 인덱스가 -2인 경우는 리스트의 뒤에서 두번째 원소가 출력
print("alphabets[-2]:", alphabets[-2])

# 리스트에서 원하는 원소를 가져와서 계산
print("alphabets[7] + alphabets[8] =", alphabets[7] + alphabets[8])

# 출력
# len(alphabets):  26
# alphabets[0]:  a
# alphabets[4]: e
# alphabets[-1]: z
# alphabets[-2]: y
# alphabets[7] + alphabets[8] = hi

print("\n*************************************indexing-add*************************************")
num_list = [1, 2, 3]

# 리스트에 append() 함수로 원소 추가
num_list.append(4)
print(num_list)
# 출력: [1, 2, 3, 4]

print("\n*************************************indexing-update*************************************")
# 리스트에 append() 함수로 원소 수정
num_list[-1] = 5  # num_list 리스트의 마지막 원소를 5로 변경
print(num_list)
# 출력: [1, 2, 3, 5]

num_list[3] = 6  # num_list 리스트의 세번째 원소를 5로 변경
print(num_list)
# 출력: [1, 2, 3, 6]


print("\n*************************************indexing-delete*************************************")
# 리스트에서 원소 삭제
# clear(): 리스트에 있는 모든 원소 삭제
num_list = [1, 2, 3]
num_list.clear()
print(num_list)
# 출력: []

# pop(): 인덱스를 통해 원소를 삭제하고 반환
num_list = [1, 2, 3]
a = num_list.pop(0)  # index가 0인 원소 삭제 후 반환
print(a)
print(num_list)
# a = num_list.pop(100)       # 인덱스 범위 외의 값으로 삭제할 경우 에러 발생 -> IndexError: pop index out of range
# 출력
# 1
# [2, 3]


# remove(): 값이 일치하는 원소르 삭제
num_list = [1, 2, 3]
num_list.remove(2)
print(num_list)
# num_list.remove(100)      # 존재하지 않는 값을 삭제하려 할 경우 에러 발생 -> ValueError: list.remove(x): x not in list
# 출력: [1, 3]

# del statement
num_list = [1, 2, 3]
del num_list[-1]  # index가 -1인 (뒤에서 첫번째) 원소를 삭제
print(num_list)
# 출력: [1, 2]

print("\n*************************************slicing*************************************")
# 리스트[가져올_원소의_첫번째_인덱스:가져올_원소의_마지막_인덱스 + 1]
num_list = [1, 2, 3, 4, 5]

print(num_list[2:])  # 인덱스 2번째부터 끝까지의 원소들
# 출력: [3, 4, 5]
print(num_list[2:3])  # 인덱스 2번째부터 3 이전까지의 원소들
# 출력: [3]
print(num_list[2:-1])  # 인덱스 2번째부터 뒤에서 첫번째 이전까지의 원소들
# 출력: [3, 4]
print(num_list[:-1])  # 처음부터 인덱스 뒤에서 첫번째 이전까지의 원소들
# 출력: [1, 2, 3, 4]
print(num_list[-3:])  # 인덱스 뒤에서 세번째부터 마지막까지의 원소들
# 출력: [3, 4, 5]

print("\n*************************************slicing-delete*************************************")
num_list = [1, 2, 3, 4, 5, 6]
del num_list[3:]  # index가 3인 원소부터 마지막 원소까지 삭제
print(num_list)
# 출력: [1, 2, 3]

