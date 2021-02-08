챕터 별 페이지 내에서 ctrl + f 로 키워드 검색 가능하시면 빠른 검색 가능합니다. 

지속적으로 업데이트 될 예정이며, 모든 질문은 class101 게시판을 이용해 주시기 바랍니다. 

[TOC] 
  

## range()

`range()` 함수는 연속된 숫자를 자동으로 만들어 줍니다.
리스트 안에 연속되는 수를 넣어주고 싶을 때 사용하며 for문과 사용되는 경우가 많습니다.
기본구조는 `range(start_value, end_value, step)` 과 같습니다.

### range(value)

```{.python}
# 하나의 인자만 주어졌을 때, 0부터 인자값-1 까지의 수를 만들어준다
num_list = list(range(10))          # 0부터 9 (10-1)까지의 수
print(num_list)
# 출력값: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### range(start_value, end_value)

```{.python}
# 두 개의 인자를 주었을 때, 첫인자부터 두번째인자-1 까지의 만큼의 수를 만들어준다
num_list = list(range(2, 5))        # 2부터 4 (5-1)까지의 수
print(num_list)
# 출력값: [2, 3, 4]
```
### range(statr_value, end_value, step)

```{.python}
# 두 개의 인자를 주었을 때, 첫인자부터 두번째인자-1 까지의 만큼의 수를 만들어준다
num_list = list(range(1, 11, 2))    # 1부터 10까지의 수를 2의 간격으로 생성
print(num_list)
# 출력값: [1, 3, 5, 7, 9]
```

## 기본문

for문은 반복문으로 Sequence (반복 가능한 형식: 리스트, 문자열 등)를 반복 연산할 때 사용됩니다.  
for 문은 Sequence 원소를 차례대로 한 개씩 받아 오며,   
모든 원소를 받아 올 때까지 반복됩니다.  
   
기본 구조   

```{.python}
for 변수 in Sequence(list, tuple, range ....) : 
    수행할 문장 
```

예시 1 - [range(value)](https://wikidocs.net/103749#rangevalue)을 통해 0~9까지 출력 + [List](https://wikidocs.net/103746)

```{.python}
num_list = []       # 빈 리스트 생성
print(num_list)
for i in range(10):
    num_list.append(i)
print(num_list)
# 출력
# []
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

예시 2 - [range(start_value, end_value)](https://wikidocs.net/103749#rangestart_value-end_value)을 통해 0~9까지 출력 + [List](https://wikidocs.net/103746)

```{.python}
num_list = []       # 빈 리스트 생성
print(num_list)
for j in range(2, 10):
    num_list.append(j)
print(num_list)
# 출력
# []
# [2, 3, 4, 5, 6, 7, 8, 9]
```
## 중첩문
중첩문을 쓰기 위해서 들여쓰기를 한번 더 해줘야 합니다.

기본 구조   

```{.python}
for 변수 1 in Sequence 1(list, tuple, range ....) : 
    수행할 문장 1
    for 변수 2 in Sequence 2(list, tuple, range....):
        수행할 문장 2
```

예시 1 - 날짜 출력
```{.python}
date_list = [('2020', '1', '1'), ('2020', '1', '2'), ('2020', '1', '3')]
for i in range(len(date_list)):             # date_list의 길이만큼 for문 돌기
    date = ''                               # 새로운 date을 출력하기 위해서 reset
    for j in range(len(date_list[i])):      # date_list 리스트 안의 tuple의 길이만큼 for문 돌기
        date += date_list[i][j] + ' '
    print(date)
# 출력
# 2020 1 1 
# 2020 1 2 
# 2020 1 3 
```

## break, continue, pass, else

`break`는 for 또는 while 문 실행을 중지하고 빠져나가도록 해줍니다.  
```{.python}
num_list = [3, 4, 2, 0, 5, 9]

for i in num_list:
    if i == 0:
        break
    print(f"12 나누기 {i} = {12 / i}")
# 출력
# 12 나누기 3 = 4.0
# 12 나누기 4 = 3.0
# 12 나누기 2 = 6.0
```

`continue`는 현재 loop를 건너뛰고 다음 loop를 실행합니다.  
`pass`는 아무것도 하지 않습니다. (주로 try: except:문에서 사용)   
```{.python}
for i in range(1, 5):
    if i == 3:
        continue        # i가 3인 경우 현재 루프를 건너뛰므로 i가 3일 때는 출력하지 않는다.
    print(i)
# 출력값: 1, 2, 4

for i in range(1, 5):
    if i == 3:
        pass            # i가 3인 경우 pass -> 코드 실행에 아무런 영향을 주지 않는다.
    print(i)
# 출력값: 1, 2, 3, 4
```

`for`문안에서 `break`문이 실행이 되지 않았을 시 루프를 다 돈 후 `else`문을 실행합니다.  
`break`로 인한 `for`문 종료 후에는 `else`문이 실행되지 않습니다.
```{.python}
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
```