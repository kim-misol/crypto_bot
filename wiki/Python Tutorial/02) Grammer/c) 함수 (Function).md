챕터 별 페이지 내에서 ctrl + f 로 키워드 검색 가능하시면 빠른 검색 가능합니다. 

지속적으로 업데이트 될 예정이며, 모든 질문은 class101 게시판을 이용해 주시기 바랍니다. 

[TOC]

  
# 함수 def()

함수의 def는 define의 약자입니다.   
효과적으로 사용하고 코드의 가독성을 높이기 위해 반복되는 코드를 함수로 정의하고 정의된 함수를 호출하여 사용합니다.    
```{.python}
def 함수이름 (parameter1,...):        # 함수 이름은 중복 불가, parameter는 입력값 
	실행할 명령1                      # 함수 내에서는 tab 키로 한 칸 띄우기 (들여쓰기)   
    	.
        .
    ruturn 결과                       # 결과값을 출력
       
함수이름(parameter1,...)              # 함수는 호출하기 이전에 정의 필수
                                     # 함수를 호출할 때는 함수 이름과 "()" 를 붙여서 사용
                                     # 함수는 호출되기 전까지는 구동하지 않는다
```
  
  
### 함수 내에서 결과값 출력
```{.python}
def sum(a, b):
    res = a + b
    print(res)

print("함수 호출 전")
sum(1, 2)         
print("함수 호출 후")

# 출력 결과:
# 함수 호출 전
# 3
# 함수 호출 후
```
   
     
### 함수 내에서 결과값 리턴하여 함수 호출 후에 출력
```{.python}
def sum(a, b):
    return a + b

print("함수 호출 전")
result = sum(1, 2)         
print("함수 호출 후")
print(result)

# 출력 결과:
# 함수 호출 전
# 함수 호출 후
# 3
```

# 매개변수(Paramete), 인자 (Argument)

Parameter: 매개변수 - 함수와 메서드 입력 변수(Variable) 명
Argument: 인자 - 함수와 메서드의 입력 값(Value)  
  
### 기본적인 형태
```{.python}
def sum(a, b):                                # a, b: Parameter 함수정의 부분에 나열
  return a + b

result = sum(1, 2)                            # 1, 2: argument 함수 호출시 전달되는 실제 값
```
    
    
### Keyword Argument 

#### 키워드 인자
parameter에 맞추어 값을 전해줍니다.    
paramter 순서가 변경되어도 됩니다.  
```{.python}
def print_date(month, day):            
  print(f"{month}월 {day}일")

print_date(month=12, day=31)                   # 12월 31일
```
  
  
#### Parameter default 값 정의  
defaut 값이 정의된 parameter는 함수 호출 시 인자를 넘겨주지 않아도 됩니다.
```{.python}
def print_date(month=1, day=1):
    print(f"{month}월 {day}일")

print_date(month=12, day=31)                   # 12월 31일
print_date()                                   # 1월 1일
```

#### keyword argument와 default 값 정의 혼합  
default 값이 정의 되지 않은 parameter 부터 작성 후 default 값이 정의된 parameter가 와야합니다.
```{.python}
def print_date(month, day=1):
    print(f"{month}월 {day}일")

print_date(month=12, day=31)                    # 12월 31일
```

```{.python}
def print_date(month, day=1):
    print(f"{month}월 {day}일")

print_date(month=12)  # 12월 1일
```
`print_date()`함수에 `day=1`이라는 default 값이 정의되어 있어 month 인자에 대한 값만 전달해주면 `day`는 default 값을 출력합니다.

```{.python}
def print_date(month, day=1):
    print(f"{month}월 {day}일")

print_date()                                    # TypeError: print_date() missing 1 required positional argument: 'month'
```
위의 코드와 같이 `print_date()` 함수 호출 시 `month` 인자 값을 넣어주지 않으면 `missing 1 required positional argument` 에러가 발생합니다.

```{.python}
def print_date(day=1, month):                   # SyntaxError: non-default argument follows default argument
    print(f"{month}월 {day}일")

print_date(month=12, day=31)  
```
keyword argument와 default 값 정의 혼합하여 사용할 때에는 반드시 default 값이 정의 되지 않은 parameter 부터 작성 후 default 값이 정의된 parameter가 와야하는데, 순서를 지키지 않을 경우 에러가 발생합니다.