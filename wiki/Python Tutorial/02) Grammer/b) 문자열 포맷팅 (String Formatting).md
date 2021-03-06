챕터 별 페이지 내에서 ctrl + f 로 키워드 검색 가능하시면 빠른 검색 가능합니다. 

지속적으로 업데이트 될 예정이며, 모든 질문은 class101 게시판을 이용해 주시기 바랍니다. 

[TOC]

## f-Strings
python 3.6 버전 이후부터 사용 가능한 문자열 포맷 (strings format) 방법입니다.  
`'{}'.format()`과 같은 기능이지만 코드가 깔끔해져 비교적 가독성이 높습니다.  
먼저 변수에 값을 대입한 후, 중괄호{}에 변수명을 넣는다. 
이때 중괄호가 들어가는 코드 처음에 f를 입력해야한다.
대문자 F도 가능하다.

```{.python}
year = 2021
month = 1
day = 1
day_of_week = "금요일"

# 쌍타옴표 앞에 f 입력
print(f"{year}년 {month}월 {day}일은 {day_of_week}입니다.")    # 2021년 1월 1일은 금요일입니다.
# 대문자 F도 가능
print(F"{year}년 {month}월 {day}일은 {day_of_week}입니다.")    # 2021년 1월 1일은 금요일입니다.
```
`{year}`이 변수 `year`의 값인 **2021**,  
`{month}`이 변수 `month`의 값인 **1**,  
`{day}`이 변수 `day`의 값인 **1**,  
`{day_of_week}`이 변수 `day_of_week`의 값인 **금요일**로 치환  

작은따옴표 3개(''') 또는 큰따옴표 3개(""")를 사용하면 여러줄인 문자열을 변수에 대입할 수 있습니다.

```{.python}
multi_line = """가나다
라마바
사아자"""

print(f"{multi_line}")
# 출력
# 가나다
# 라마바
# 사아자
```
## '{}'.format()

`'{}'.format()` 문자열 포맷 (strings format) 연산을 수행합니다.    
이 메서드가 호출되는 문자열은 중괄호 `{}` 로 구분된 치환 필드를 포함하며, 각 치환 필드는 **인덱스**나 **키워드 인자**의 이름을 가집니다.  
각 치환 필드에 해당 인자의 값을 넣어줍니다.  

### 치환필드에 인덱스

#### 숫자형 치환
```{.python}
a = "{0}년 {1}월 {2}일".format(2021, 1, 1)
print(a)
# 출력: 2021년 1월 1일
```
`{0}`이 `format()`함수에 0번째 인덱스인 **2021**,  
`{1}`이 `format()`함수에 첫번째 인덱스인 **1**,  
`{2}`이 `format()`함수에 두번째 인덱스인 **1**로 치환


```{.python}
b = "{2}년 {1}월 {0}일".format(1, 1, 2021)
print(b)
# 출력: 2021년 1월 1일
```
`{2}`이 `format()`함수에 두번째 인덱스인 **2021**,  
`{1}`이 `format()`함수에 첫번째 인덱스인 **1**,  
`{0}`이 `format()`함수에 0번째 인덱스인 **1**로 치환

#### 문자열 치환
```{.python}
c = "{0}년 {1}월 {2}일 {3}".format(2021, 1, 1, '금요일')
print(c)
# 출력: 2021년 1월 1일 금요일
```
`{0}`이 `format()`함수에 0번째 인덱스인 **2021**,  
`{1}`이 `format()`함수에 첫번째 인덱스인 **1**,  
`{2}`이 `format()`함수에 두번째 인덱스인 **1**,  
`{3}`이 `format()`함수에 세번째 인덱스인 **금요일**로 치환

#### 변수 치환
```{.python}
year = 2021
month = 1
day = 1
day_of_week = '금요일'
d = "{0}년 {1}월 {2}일 {3}".format(year, month, day, day_of_week)
print(d)
# 출력: 2021년 1월 1일 금요일
```
`{0}`이 `format()`함수에 0번째 인덱스, 변수 `year`의 값인 **2021**,  
`{1}`이 `format()`함수에 첫번째 인덱스, 변수 `month`의 값인 **1**,  
`{2}`이 `format()`함수에 두번째 인덱스, 변수 `day`의 값인 **1**,  
`{3}`이 `format()`함수에 세번째 인덱스, 변수 `day_of_week`의 값인 **금요일**로 치환  

### 치환필드에 키워드 인자
```{.python}
e = "{year}년 {month}월 {day}일 {day_of_week}".format(year=2021, month=1, day=1, day_of_week='금요일')
print(e)
# 출력: 2021년 1월 1일 금요일
```
`{year}`이 `format()`함수에 키워드 인자 `year`의 값인 **2021**,  
`{month}`이 `format()`함수에 키워드 인자 `month`의 값인 **1**,  
`{day}`이 `format()`함수에 키워드 인자 `day`의 값인 **1**,  
`{day_of_week}`이 `format()`함수에 키워드 인자 `day_of_week`의 값인 **금요일**로 치환  
