챕터 별 페이지 내에서 ctrl + f 로 키워드 검색 가능하시면 빠른 검색 가능합니다. 

지속적으로 업데이트 될 예정이며, 모든 질문은 class101 게시판을 이용해 주시기 바랍니다. 

[TOC]

## 조건

[ bool 자료형 ]  
    `True`  : 참  
    `False` : 거짓  
    
[ 비교연산자 ] : 
비교 결과가 맞으면 `True`, 아니면 `False`

  `x < y`	x가 y보다 작으면 `True` / 작지 않으면 `False`  
  `x > y`	x가 y보다 크면 `True` / 크지 않으면 `False`  
  `x == y`	x와 y가 같으면 `True` / 같지 않으면 `False`  
  `x != y`	x와 y가 같지 않으면 `True` / 같으면 `False`  
  `x >= y`	x가 y보다 크거나 같으면 `True` / 작으면 `False`  
  `x <= y`	x가 y보다 작거나 같으면 `True` / 크면 `False`
    
## if 조건문

조건이 참인지 거짓인지 판단하여 참인 경우 if 문 안의 코드를 실행합니다.  
if문 기본구조
```{.python}
if 조건:                    
	실행할 명령1
       .
       .
```

예시 1 - 조건이 맞는 경우에만 if 코드블럭 실행

```{.python}
# 조건이 참이면 명령1을 실행한 후에 명령2 실행
# 조건이 거짓이면 바로 명령2 실행
a = 10             
b = 8             

if a > b:
    print("a는 b 보다 크다.")       
    
# a는 b 보다 크다.

if a <= b:
    print("a는 b 보다 크다.")       
```

## if ~ else 조건문

if ~ else문 기본구조
```{.python}
if 조건:           # 조건이 참이면 명령1 실행
	실행할 명령1
else:
	실행할 명령2  # 조건이 거짓이면 명령2 실행
```

예시 1 - 조건이 맞으면 if 코드블럭 실행하고, 맞지 않는다면 else 코드블럭을 실행

```{.python}
a = 10
b = 5

if a == b:
    print("같다.")
else:
    print("다르다")
# 다르다
```

## if ~ elif ~ else 조건문

if ~ else문 기본구조
```{.python}
if 조건:           # 조건이 참이면 명령1 실행
	실행할 명령1
else:
	실행할 명령2  # 조건이 거짓이면 명령2 실행
```

예시 1 - `a == b` 조건이 맞으면 if 코드블럭 실행하고, `a > b` 조건이 맞으면 elif 코드블럭을 실행하고,  
만족하는 경우가 없다면 else 코드 블럭을 실행

```{.python}
a = 10
b = 5

if a == b:
    print("같다.")
elif a > b:
    print("a가 b보다 크다")
else:
    print("a가 b보다 작다")
    
# a가 b보다 크다
```

## if 중첩문 (Nested if)

nested if 기본구조
```{.python}
if 조건1:
	if 조건2:
    	실행할 명령1     # 조건1, 조건2 모두 True 인 경우
    else:
    	실행할 명령2     # 조건1은 True, 조건2가 False 인 경우
else: 
    실행할 명령3      #조건1이 False 인 경우
```
예시 1 
```{.python}
a = 10
b = 5

if type(a) is int:
    if type(b) is int:
        print("a, b 모두 정수이다")
    else:
        print("a만 정수이다")
else:
    if type(b) is int:
        print("b만 모두 정수이다")
    else:
        print("a, b 정수가 아니다")

# a, b 모두 정수이다


b = "5"   # 변수 b를 string 타입으로 변경
if type(a) is int:
    if type(b) is int:
        print("a, b 모두 정수이다")
    else:
        print("a만 정수이다")
else:
    if type(b) is int:
        print("b만 모두 정수이다")
    else:
        print("a, b 정수가 아니다")

# a만 정수이다
```
