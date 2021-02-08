챕터 별 페이지 내에서 ctrl + f 로 키워드 검색 가능하시면 빠른 검색 가능합니다. 

지속적으로 업데이트 될 예정이며, 모든 질문은 class101 게시판을 이용해 주시기 바랍니다. 

[TOC] 

# 리스트 (list)
리스트는 가변 시퀀스 (순서가 있는 수정 가능한 객체의 집합)으로 수정, 삭제, 추가가 가능합니다.  
list 는 `[]` 대괄호로 감싸지고 내부 원소는 `,` 쉼표로 구분됩니다.
  
|  |  |  
| -------- | -------- | 
| 리스트 (List)     | 값을 나열해서 저장하는 것     | 
| 원소     | 리스트에 저장된 각각의 값들     |
| 인덱스 (index, 색인)     | 원소의 위치 값 (**index는 0부터 시작**)     |



# 인덱싱 (indexing)
인덱싱 (indexing)은 문자열, 리스트 또는 튜플 등에 정해진 주소를 사용해서 원하는 값을 가져오는 것을 말합니다.

리스트 `a = ['p', 'r', 'o', 'b', 'e']`    
![](https://wikidocs.net/images/page/103746/python_list_index.png)
```{.python}
a = ['p', 'r', 'o', 'b', 'e']
print(a[0])     # a의 인덱스가 0인 원소(값) 출력: p
print(a[-5])    # a의 뒤에서부터 다섯번째인 원소(값) 출력: p

print(a[1])     # a의 인덱스가 1인 원소(값) 출력: r  
print(a[-4])    # a의 뒤에서부터 네번째인 원소(값) 출력: r

print(a[2])     # a의 인덱스가 2인 원소(값) 출력: o
print(a[-3])    # a의 뒤에서부터 세번째인 원소(값) 출력: o

print(a[3])     # a의 인덱스가 3인 원소(값) 출력: b
print(a[-2])    # a의 뒤에서부터 두번째인 원소(값) 출력: b

print(a[4])     # a의 인덱스가 4인 원소(값) 출력: e
print(a[-1])    # a의 뒤에서부터 첫번째인 원소(값) 출력: e

# 리스트에서 원하는 원소를 가져와서 계산 
print(a[0] + a[1] + a[2])  # 출력: pro
```

##  원소 추가 
#### append()
```{.python}
num_list = [1, 2, 3]

# 리스트에 append() 함수로 원소 추가
num_list.append(4)
print(num_list)
# 출력: [1, 2, 3, 4]
```

#### list + list 
```{.python}
# 리스트끼리 더하기
result = num_list + num_list
print(result)
# 출력: [1, 2, 3, 1, 2, 3]
```

## 원소 수정
```{.python}
# 리스트에 append() 함수로 원소 수정
num_list[-1] = 5    # num_list 리스트의 마지막 원소를 5로 변경
print(num_list)
# 출력: [1, 2, 3, 5]

num_list[3] = 6    # num_list 리스트의 세번째 원소를 5로 변경
print(num_list)
# 출력: [1, 2, 3, 6]
```

## 원소 삭제 
 `clear()`, `pop()`, `remove()`, `del statement`의 방법들을 사용하여 list의 원소(들)을 삭제할 수 있습니다.  
 
#### clear()
```{.python}
# clear(): 리스트에 있는 모든 원소 삭제
num_list = [1, 2, 3]
num_list.clear()
print(num_list)
# 출력: []
```

#### pop()
```{.python}
# pop(): 인덱스를 통해 원소를 삭제하고 반환
num_list = [1, 2, 3]
a = num_list.pop(0)           # index가 0인 원소 삭제 후 반환
print(a)
print(num_list)
# a = num_list.pop(100)       # 인덱스 범위 외의 값으로 삭제할 경우 에러 발생 -> IndexError: pop index out of range
# 출력
# 1
# [2, 3]
```

#### remove()
```{.python}
# remove(): 값이 일치하는 원소르 삭제
num_list = [1, 2, 3]
num_list.remove(2)
print(num_list)
# num_list.remove(100)        # 존재하지 않는 값을 삭제하려 할 경우 에러 발생 -> ValueError: list.remove(x): x not in list
# 출력: [1, 3]
```

#### del statement
```{.python}
# del statement
num_list = [1, 2, 3]
del num_list[-1]  	          # index가 -1인 (뒤에서 첫번째) 원소를 삭제
print(num_list)
# 출력: [1, 2]
```

# 슬라이싱 (slicing)
슬라이싱 (slicing)은 문자열, 리스트, 튜플 등을 특정 인덱스 구간에 맞춰 잘라 가져오는 것을 말합니다.  
리스트[가져올_원소의_첫번째_인덱스:가져올_원소의_마지막_인덱스** + 1**]

```{.python}
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

# del statement로 리스트 슬라이싱하여 삭제
num_list = [1, 2, 3, 4, 5, 6]
del num_list[3:]  # index가 3인 원소부터 마지막 원소까지 삭제
print(num_list)
# 출력: [1, 2, 3]
```
# list와 tuple 비교

| 리스트 (list) | 튜플 (tuple) |
| -------- | -------- | 
| 원소 생성, 삭제, 수정 가능    | 원소 생성, 삭제, 수정 불가능      | 
| 원소를 `[]`으로 감싼다    | 원소를 `()`으로 감싸며, 괄호를 사용하지 않아도 표현이 가능하다      |  
| 빈 리스트 생성 시 `a = []`    | 빈 튜플 생성 시 `a = ()`      | 


