챕터 별 페이지 내에서 ctrl + f 로 키워드 검색 가능하시면 빠른 검색 가능합니다. 

지속적으로 업데이트 될 예정이며, 모든 질문은 class101 게시판을 이용해 주시기 바랍니다. 


[TOC]

# 클래스, 객체

| 클래스(class) | 객체(object) | 
| -------- | -------- | 
| 붕어빵틀     | 팥붕어빵, 슈크림붕어빵, 고구마붕어빵 ...    | 
| 자동차 설계도     | 자동차 ...    | 
| CandleType     | 1분봉, 3분봉, 5분봉, 10분봉 ...   | 

### 클래스
**클래스**는 변수 (variable)과 함수 (def)를 통해 새로운 **객체**를 생성하여 반복되는 코드를 최소화 하면서 쉽게 호출하여 사용할 수 있습니다.  
**클래스**는 객체를 찍어내기 위한 설계도이고  
만들어진 **객체**는 각각 고유한 기능 또는 성격을 가지게 됩니다.  
붕어빵틀을 클래스라고 한다면, 팥붕어빵, 슈크림붕어빵, 고구마붕어빵 등은 객체라고 표현할 수 있습니다.  

### 객체
객체는 `객체이름 = 클래스이름()` 으로 생성되며, 객체이름.변수, 객체이름.메서드()의 형식으로 객체가 지닌 변수와 메서드를 사용할 수 있습니다.

### 클래스의 멤버
클래스 내부에 포함되는 변수, 함수를 클래스의 멤버라고 부릅니다.

### 클래스의 함수 (메서드)
클래스 내부에 포함되는 함수를 메서드라고 부릅니다.

### 매직 메서드 (masic method)
`masic method`는 클래스안에 정의할 수 있는 스페셜 메서드로 클래스를 파이썬의 빌트인 타입과 같은 작동을 하게 해줍니다.
`__init__`이나 `__repr__`과 같이 메소드 이름 앞뒤에 **더블 언더스코어**("\__: 던더, dunder, Double Underscore")를 붙입니다

`__init__`: 'Initializer' (생성자, 클래스 이니셜라이저)의 약자로, 메소드의 첫 매개변수는 `self`를 받습니다.
클래스 객체를 생성하면서 직접적으로 `__init__`메소드를 호출하지 않지만 내부적으로는 실행되며, 인스턴스 객체 생성 시 `__init__`에 전달할 인수를 취합니다.

`__repr__`: 'Representation' (표현)의 약자로 어떤 객체의 '출력될 수 있는 표현' (printable representation)을 문자열의 형태로 반환합니다.
사용자가 이해하기 쉽게 객체의 모습을 표현해줍니다. 
-----


예시 1
```{python}
class User:                                   # User 이라는 이름을 가진 클래스를 정의
    def __init__(self, first_name, last_name):       # __init__: '생성자'(특별한 함수) 이며, 클래스의 객체를 만들 때 자동으로 실행이 된다.
        self.first_name = first_name                 # self.first_name, self.last_name를 인스턴스 변수라고 한다.
        self.last_name = last_name

    def get_full_name(self):
        return self.last_name + self.first_name      # self.last_name과 self.first_name 문자열을 더한 값을 반환

u = User("길동", "홍")
fullname = u.get_full_name("***")
print(f"first name: {u.first_name}, full name: {fullname}")

# 출력: 
# first name: 길동, full name: 홍길동
```

예시 2
```{python}
class Calculator:
    def __init__(self, first, second):
        # self.x, self.y  : 인스턴트 변수 (instance variable)
        self.x, self.y = first, second

    def add(self):
        return self.x + self.y

    def sub(self):
        return self.x - self.y

    def mul(self):
        return self.x * self.y

    def mod(self):
        return self.x / self.y

calc = Calculator(10, 5)        # calc 객체에 Calculator 클래스를 통해 인스턴트 변수 self.x에는 10과 self.y에는 5가 저장된다.
print(calc.add())               # 인스턴트 변수 self.x와 self.y를 calc 객체 안에 있는 add 메서드를 호출하여 계산
print(calc.sub())               # 인스턴트 변수 self.x와 self.y를 calc 객체 안에 있는 sub 메서드를 호출하여 계산
print(calc.mul())               # 인스턴트 변수 self.x와 self.y를 calc 객체 안에 있는 mul 메서드를 호출하여 계산
print(calc.mod())               # 인스턴트 변수 self.x와 self.y를 calc 객체 안에 있는 mod 메서드를 호출하여 계산

# 출력:
# 15
# 5
# 50
# 2.0
```

# 클래스 상속 (inheritance)

쉽게 말해서, 부모의 유산을 자식이 물려 받듯이 **클래스에서의 상속**은 부모 클래스 (Parent Class)의 **속성과 메소드**를 자식 클래스 (Child Class)가 가지게 되는 것입니다.

자

| 부모 클래스  | 자식 클래스 |
| -------- | -------- |
| Parent Class     | Child Class     | 
| 베이스 클래스    | 파생 클래스     | 
| 속성과 메서드를 물려준다     | 속성과 메서드를 물려받는다     | 

파생 클래스의 객체가 만들어질 때, 베이스 클래스가 기억되며,
파생 클래스는 베이스 클래스의 메서드들을 재정의할 수 있습니다. 

```{.python}
class User:
    def __init__(self, first_name, last_name):
        self.first_name = first_name        # 3)
        self.last_name = last_name          # 4)

    def get_full_name(self):
        return self.last_name + self.first_name         # 8)


class UserDetail(User):
    def __init__(self, first_name, last_name, email):
        User.__init__(self, first_name, last_name)      # 2)
        self.email = email                              # 5)

    def print_user_info(self):
        full_name = User.get_full_name(self)                        # 7)
        print(f"{full_name}의 이메일 주소는 {self.email}입니다.")     # 9)


gildong = UserDetail('길동', '홍', 'example@gmail.com')  # 1)
gildong.print_user_info()                               # 6)

nada = UserDetail('나다', '가', 'example1@gmail.com')  # 10)
nada.print_user_info()                               11)

# 출력:
# 홍길동의 이메일 주소는 example@gmail.com입니다.
# 가나다의 이메일 주소는 example1@gmail.com입니다.
```

1) `UserDetail` 클래스의 객체 생성  
2) `UserDetail` 클래스의 masic method `__init__`에서 부모클래스인 `User` 클래스의 `__init__`를 호출  
3) `first_name` 인자를 전달받아 `self.first_name` 정의  
4) `last_name` 인자를 전달받아 `self.last_name` 정의  
5) `email` 인자를 전달받아 `self.email` 정의  
6) `gildong` 인스턴스의 `print_user_info()` 함수 호출  
7) 부모 클래스인 `User` 클래스의 `get_full_name(self)` 함수를 호출  
8) `get_full_name(self)` 함수로부터 `self.last_name + self.first_name` 문자열 연산되어 반환된 값을 `full_name` 에 저장  
9) f-string을 이용하여 `full_name`과 `self.email` 값을 출력  
10) 1)부터 5)를 반복하여 `nada` 인스턴스 생성    
11) 7)부터 9)를 반복하여 '가나다의 이메일 주소는 example1@gmail.com입니다.' 출력  
12) 프로그램 실행 완료    




