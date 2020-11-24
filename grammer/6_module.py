"""
module: 여러 함수와 변수, 클래스를 포함한 Python 파일
package: module 묶음
library: package 묶음

import ? -> 패키지나 모듈을 가져올 때 사용
Python에는 기본적으로 내장되어있는 라이브러리가 있다. Python Standard Library https://docs.python.org/3/library/
실제로 많이 쓰고 유용한 것들이 많다.
"""

print("\n*************************************ex1*************************************")
'''
 제곱근 구하기
 math 라는 모듈에 뭐가 있는지 궁금하면? ctrl + 클릭 -> 
 Python에는 기본적인 수학과 관련된 함수들을 math라는 모듈에 만들어 두었다.
 제곱근을 구하는 방법은 sqrt라는 함수를 이용하면 된다.
 
 암기 x!! 구글에 'python 제곱근 구하기 모듈' 을 검색하면 모듈 사용 방법이 나와 있다. 
'''

import math

# math 라는 모듈에서 sqrt 라는 함수를 선택한 것.
a1 = math.sqrt(9)
print(a1)

print("\n*************************************ex2*************************************")
# math 모듈에는 수많은 함수들이 미리 정의가 되어있는데 그 중에 sqrt라는 함수만 사용하고 싶은 경우
# from 모듈 import 변수, 함수, 클래스

from math import sqrt

# sqrt 앞에 math. 을 쓰지 않아도 바로 사용 가능
a2 = sqrt(9)
print(a2)

print("\n*************************************ex3*************************************")
# math 모듈에는 수많은 함수들이 미리 정의가 되어있는데 그 중 모든 함수, 변수, 클래스 를 참조 하고 싶을 경우(* 쓰는 건 최대한 지양해야함)

# from 모듈 import *
from math import *

# sqrt 앞에 math. 을 쓰지 않아도 바로 사용 가능
a3 = sqrt(9)
print(a3)

print("\n*************************************ex4*************************************")
# from 모듈 import 변수 as 이름
# from 모듈 import 함수 as 이름
# from 모듈 import 클래스 as 이름
from math import sqrt as sq

a4 = sq(9)
print(a4)

print("\n*************************************ex5*************************************")
# 패키지 만들기 실습(package_test.py 파일 및 my_package 패키지 생성 후 실습)
# from 패키지 import 모듈 (ex. from my_package import sum)


# from 패키지.모듈 import 함수, 클래스 (ex. from my_package.sum import sum_ab)

# __name__ ? => 현재 모듈의 이름을 담고 있는 내장 변수


# 이어지는 수업에서 설명
print("\n*************************************ex6*************************************")
'''
# 패키지 다운로드 방법
# pip : 파이썬으로 작성된 패키지 소프트웨어를 설치하거나 관리하는 패키지 관리 시스템
# 1. 아래 terminal 탭 클릭 후 -> pip install '원하는 패키지 이름'   (ex. pip install pymysql)
# 2. 설치 된 모듈 리스트 : ctrl + alt + s (file -> setting ->project:bot -> python interpreter)

/* 패키지 설치 */
# pip install <package name>

/* 설치된 패키지 업그레이드 */
# pip install --upgrade <package name>

/* 패키지 제거 */
# pip uninstall <package name>

/* 설치된 패키지 파일 목록 */
# pip list


'''
