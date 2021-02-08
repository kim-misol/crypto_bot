챕터 별 페이지 내에서 ctrl + f 로 키워드 검색 가능하시면 빠른 검색 가능합니다. 

지속적으로 업데이트 될 예정이며, 모든 질문은 class101 게시판을 이용해 주시기 바랍니다. 

[TOC] 

## while문 조건
while loop은 조건이 **참**인 동안 실행됩니다.  
for loop과 다르게 조건으로 문자열, 리스트, 튜플과 같은 시퀀스 외에도 참과 거짓을 가지는 모든 조건절이 올 수 있습니다.


| 변수 타입 (Variable Type) | 참 (True) | 거짓 (False) |
| -------- | -------- | -------- |
| Boolean     | True     | False     |
| 숫자형 (정수)     | 0을 제외한 모든 정수     | 0     |
| 문자열     | 길이가 0이 아닌 string     | ''     |
| 리스트 (list)     | 길이가 0이 아닌 list     | []     |
| 튜플 (tuple)     | 길이가 0이 아닌 tuple     | ()     |

```{.python}
# Boolean type
if True:
    print('True: 참')
if False:
    print('False: 거짓이므로 출력되지 않는다')

# int type
if 10:
    print('10: 참')
if -10:
    print('-10: 참')
if 0:
    print('0: 거짓이므로 출력되지 않는다')

# str type
if "a":
    print('"a": 참')
if "":
    print('"": 거짓이므로 출력되지 않는다')

# list type
if [1, 2, 3]:
    print('[1,2,3]: 참')
if []:
    print('[]: 거짓이므로 출력되지 않는다')

# tuple type
if (1, 2, 3):
    print('(1,2,3): 참')
if ():
    print('(): 거짓이므로 출력되지 않는다')
# 출력
# True: 참
# 10: 참
# -10: 참
# "a": 참
# [1, 2, 3]: 참
# (1, 2, 3): 참
```

## continue, pass, break, else
`break`는 [for문](https://wikidocs.net/103749#break-continue-pass-else) 또는 while 문 실행을 중지하고 빠져나가도록 해줍니다.
`continue`는 현재 loop를 건너뛰고 다음 loop를 실행합니다.
`pass`는 아무것도 하지 않습니다. (주로 try: except:문에서 사용)
```{.python}
i = 0
while True:
    i += 1
    if i == 2:
        pass            # i가 2일 때도 아무런 변화 없이 loop를 실행
    elif i == 3:
        continue        # i가 3인 loop를 건너뛰고 다음 loop 실행 ("i = 3" 출력을 하지 않는다.)
    elif i > 7:
        break           # i가 7 초과인 경우, while loop을 정지하고 나간다.
    print("i =", i)
# 출력
# i = 1
# i = 2
# i = 4
# i = 5
# i = 6
# i = 7
```

`while` 문의 조건과 맞지 않아 loop를 종료할 때 `else`문을 실행합니다.  
`break`로 인한 `while`문 종료에서는 `else`문이 실행되지 않습니다.
```{.python}
i = 0
while i < 3:
    print("i =", i)
    i += 1
else:               # while이 끝난 후 else로 들어와서 아래 메세지 출력
    print("while loop END")
# 출력
# i = 0
# i = 1
# i = 2
# while loop END
```
  
 