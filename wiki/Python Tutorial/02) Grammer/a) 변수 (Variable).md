챕터 별 페이지 내에서 ctrl + f 로 키워드 검색 가능하시면 빠른 검색 가능합니다. 

지속적으로 업데이트 될 예정이며, 모든 질문은 class101 게시판을 이용해 주시기 바랍니다. 

[TOC]


| 영  | 한 | 설명 |
| -------- | -------- | -------- |
| Variable     | 변수     | 숫자, 텍스트 등을 저장 할 수 있는 방     |
| Variable Name     | 변수명     | 변수명은 알파벳 또는 언더바 (_)로 시작     |
| Value     | 변수 값     | 변수에 값을 저장할 때는 "=" 를 사용     |


```{.python}
date = "20210101"
```

위의 date은 변수 (variable)이며, 변수 값 (value)는 "20210101" 입니다.  
"=" 기호를 사용하여 변수에 값을 저장할 수 있습니다.
  
또한 변수에 저장된 값을 변경하기 위해서는 다른 값을 "=" 기호를 사용하여 재저장하면 됩니다. 

```{.python}
# 기존 값
date = "20210101"
# 변경된 값
date = "20201231"

print(date)
# 20201231
```

## 변수명 유의할 점 (Naming)

변수명은 특정 convention을 제외하고 자유롭게 지정할 수 있습니다.

- 가능한 변수명
    - Alpabet (영문 또는 한글)
    - 언더바 (_)
    - 대소문자 구분가능  
    
\- Python Console -
```{.python}
>>> a = 1             # Alpabet (영문)으로 시작 또는 포함
>>> a
1
>>> _a = 1            # 언더바 (_)으로 시작 또는 포함
>>> _a
1
>>> b = 2             # 변수 값 변경 가능
>>> b = 4
>>> b
4
>>> 파이썬="Python"    # Alpabet (한글)으로 시작 또는 포함
>>> 파이썬
Python
```
- 불가능한 변수명

```{.python}
>>> ＊= 3                # 도형을 포함한 변수명은 불가능 
SyntaxError: invalid character in identifier
>>> ☆ = 1
SyntaxError: invalid character in identifier
>>> 1var = 1             # 숫자로 시작하는 변수명은 불가능
SyntaxError: invalid syntax
>>> var1 = 1             # 숫자로 끝나는 변수명은 가능
>>> var1
1
>>> variable name = 1    # 변수에 띄어쓰기 불가능
SyntaxError: invalid syntax
```

## 변수 type

### 숫자형
```{.python}
# 변수 값 숫자 (number: int, float)
a1 = 20210101
a2 = 10.5
print(a1)          # 20210101
print(type(a1))    # <class 'int'>
print(a2)          # 10.5
print(type(a2))    # <class 'float'>
```

### 문자열
```{.python}
# 변수 값 텍스트 (text: str)
b = "20210101"
print(b)          # 20210101
print(type(b))    # <class 'str'>
```

### Boolean
```{.python}
# 변수 값 Boolean (참, 거짓: bool)
is_true = True
is_false = False
print(is_true)          # True
print(is_false)         # False
print(type(is_true))    # <class 'bool'>
print(type(is_false))   # <class 'bool'>
```

### Date
```{.python}
from datetime import datetime

# 변수 값 날짜 (date)
c = datetime(2021, 1, 1)
print(c)          # 2021-01-01 00:00:00
print(type(c))    # <class 'datetime.datetime'>
```

### List
```{.python}
# 변수 값 리스트 (list)
d = ['20210101', '20210102', '20210103']
print(d)          # ['20210101', '20210102', '20210103']
print(type(d))    # <class 'list'>
print(type(d[0])) # <class 'str'>
print(d[0])       # 20210101
print(d[1])       # 20210102
print(d[2])       # 20210103
print(d[-1])      # 20210103
```


### Tuple
```{.python}
# 변수 값 튜플 (tuple)
e = (0, '20210101')
print(e)        # (0, '20210101')
print(type(e))  # <class 'tuple'>
print(e[0])     # 0
print(e[1])     # 20210101

```

### Python Document list
[숫자](https://docs.python.org/ko/3/tutorial/introduction.html#numbers)

[문자열](https://docs.python.org/ko/3/tutorial/introduction.html#strings)

[리스트](https://docs.python.org/ko/3/tutorial/introduction.html#lists)

[튜플](https://docs.python.org/ko/3/tutorial/datastructures.html#tuples-and-sequences)