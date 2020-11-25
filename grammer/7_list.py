print("\n*************************************ex1*************************************")
'''
    리스트(list)? : 값을 나열해서 저장하는 것
    원소 : 리스트에 저장 된 각각의 값들
    인덱스(index, 색인) : 위치 값
    여기서 주의할 점은 리스트의 인덱스는 항상 0부터 시작한다는 점입니다.
    즉, 리스트 a의 첫 번째 원소는 a[0]
   '''

buy_list = ['비트코인', '이더리움', '리플', '비트코인캐시', '체인링크']
# buy_list = "비트코인"  # 리스트가 아닌 string 도 동일하게 인덱싱 가능
num = 5
# buy_list의 자료형은 list  ( * 비트코인 인 경우 자료형은 str)
print("type(buy_list): ", type(buy_list))
# num 의 자료형은 int
print("type(num): ", type(num))

# buy_list 리스트 전체 출력
print("buy_list: ", buy_list)

'''
 인덱싱 : 인덱싱이라는 것은 무엇인가를 '가리킨다'는 의미입니다.
            즉, 문자열이나 리스트, 튜플 등에 번호를 부여하여 특정 위치를 가리키는 것을 말합니다. ex) A[0], A[-1]
'''
# buy_list의 0번째 인덱스의 요소(값)는 0
print("buy_list[0]: ", buy_list[0])

print("buy_list[4]:", buy_list[4])

# len은 buy_list라는 리스트의 길이를 알려주는 내장 함수
print("len(buy_list): ", len(buy_list))

# a[-1] 처럼 인덱스가 -1인 경우는 리스트의 마지막 원소가 출력
print("buy_list[-1]:", buy_list[-1])

# a[-2] 처럼 인덱스가 -2인 경우는 리스트의 뒤에서 두번째 원소가 출력
print("buy_list[-2]:", buy_list[-2])

print("buy_list[0]+buy_list[1] =", buy_list[0] + buy_list[1])

'''
슬라이싱 : 문자열, 리스트, 튜플 등을 특정 인덱스 구간에 맞춰 잘라 내는 것 
'''

# 인덱스 0부터 인덱스 3까지의 원소 출력
print("buy_list[0:4]: ", buy_list[0:4])
# 첫 인덱스(0)와 끝 인덱스는 생략가능 위 출력결과와 동일
print("buy_list[:4]: ", buy_list[:4])

# 인덱스 4 부터 끝까지
print("buy_list[4:]: ", buy_list[4:])

# 인덱스 2 부터 인덱스 5 까지
print("buy_list[2:6]: ", buy_list[2:6])

print("\n*************************************ex2*************************************")
a = ["안녕", "하세요", "저는", "홍길동", "입니다"]
date_rows = ["20201123", "20201124", "20201125", "20201126", "20201127"]
time_rows = ["0900", "0930", "1000", "1030", "1100"]
print("date_rows: ", date_rows)
print("type(date_rows): ", type(date_rows))
print("len(date_rows): ", len(date_rows))
print("date_rows[0]: ", date_rows[0])
print("date_rows[4]:", date_rows[4])
print("date_rows[-1]:", date_rows[-1])
print("date_rows[0]+time_rows[0] =", date_rows[0] + time_rows[0])
print("오름차순 정렬: ", sorted(date_rows))  # 오름차순 정렬
print("내림차순 정렬: ", sorted(date_rows, reverse=True))  # 내림차순 정렬

print("\n*************************************ex3*************************************")
# 리스트 vs 튜플 : 튜플은 리스트와 동일하지만, 수정이 불가
a = [1, 2, 3]
b = (1, 2, 3)
print(f"a[0] : {a[0]}, b[0] : {b[0]}")

# a[1] = 5 # 에러 나지 않음
# print(a)
#
# b[1] = 5 # 에러 발생
# print(b)
