# import math

from grammer.class_v2 import Calculator, power

print("\n*************************************클래스 호출하여 객체 생성*************************************")
# class_v2 파일에서 만든 Calculator 클래스의 add 메서드로 더하기
calc = Calculator(10, 5)
add_result = calc.add()
print(f"10과 5의 덧셈: {add_result}")

print("\n*************************************매서드 임포트하여 호출*************************************")
num = 4
num_power = power(num)
print(f"3의 제곱: {num_power}")

print("\n*************************************내장된 모듈 임포트하여 호출*************************************")
# import  모듈
import math     # math 라는 모듈에서 sqrt 라는 함수를 선택


num = 4
num_sqrt = math.sqrt(num)
print(f"4의 제곱근: {num_sqrt}")
# 출력: 4의 제곱근: 2.0


# from 모듈 import 특정 함수
from math import sqrt


# sqrt 앞에 math. 을 쓰지 않아도 바로 사용 가능
num = 4
num_sqrt = sqrt(num)
print(f"4의 제곱근: {num_sqrt}")
# 출력: 4의 제곱근: 2.0


# from 모듈 import 모든 함수
# math 모듈에는 수많은 함수들이 미리 정의가 되어있는데
# 그 중 모든 함수, 변수, 클래스 를 참조 하고 싶을 경우 (* 쓰는 건 지양합니다)
from math import *


# sqrt 앞에 math. 을 쓰지 않아도 바로 사용 가능
num = 4
num_sqrt = sqrt(num)
print(f"4의 제곱근: {num_sqrt}")
# 출력: 4의 제곱근: 2.0


# 모듈, 변수, 함수, 클래스를 임포트하여 원하는 이름으로 사용
from math import sqrt as sq


num = 4
num_sqrt = sq(num)
print(f"4의 제곱근: {num_sqrt}")
# 출력: 4의 제곱근: 2.0
