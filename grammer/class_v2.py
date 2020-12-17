class User:  # User 이라는 이름을 가진 클래스를 정의
    def __init__(self, name, password):  # __init__: '생성자'(특별한 함수) 라고 한다.
        # 클래스의 객체를 만들 때 자동으로 실행이 된다.
        self.name = name  # 인스턴스변수 : self.name, self.password를 인스턴스 변수라고 한다.
        self.password = password

    def check_pw_length(self, password):
        if len(password) >= 8:
            print("8자리 이상 비밀번호 입니다.")
        else:
            print("8자리 이상 비밀번호를 입력해주세요.")


u = User("타임퍼센트", "***")
u.check_pw_length("***")
print(f"name: {u.name}, password: {u.password}")


# 출력:
# 8자리 이상 비밀번호를 입력해주세요.
# name: 타임퍼센트, password: ***


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


calc = Calculator(10, 5)  # calc 객체에 Calculator 클래스를 통해 인스턴트 변수 self.x에는 10과 self.y에는 5가 저장된다.
print(calc.add())  # 인스턴트 변수 self.x와 self.y를 calc 객체 안에 있는 add 메서드를 호출하여 계산
print(calc.sub())  # 인스턴트 변수 self.x와 self.y를 calc 객체 안에 있는 sub 메서드를 호출하여 계산
print(calc.mul())  # 인스턴트 변수 self.x와 self.y를 calc 객체 안에 있는 mul 메서드를 호출하여 계산
print(calc.mod())  # 인스턴트 변수 self.x와 self.y를 calc 객체 안에 있는 mod 메서드를 호출하여 계산


# 출력:
# 15
# 5
# 50
# 2.0


def power(a):
    return a * a


print("\n************************************* 클래스 상속 예제 *************************************")


class User:
    def __init__(self, first_name, last_name):
        self.first_name = first_name  # 4)
        self.last_name = last_name  # 5)

    def get_full_name(self):
        return self.last_name + self.first_name  # 8)


class UserDetail(User):
    def __init__(self, first_name, last_name, email):
        User.__init__(self, first_name, last_name)  # 2)
        self.email = email  # 3)

    def print_user_info(self):
        full_name = User.get_full_name(self)  # 7)
        print(f"{full_name}의 이메일 주소는 {self.email}입니다.")  # 9)


u = UserDetail('길동', '홍', 'example@gmail.com')  # 1)
u.print_user_info()  # 6)

# 출력:
# 홍길동의 이메일 주소는 example@gmail.com입니다.
