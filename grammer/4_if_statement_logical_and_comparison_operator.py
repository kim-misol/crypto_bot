print("\n*************************************ex1*************************************")
'''
[ bool 자료형 ]
    True : 참
    False : 거짓
    
[ 비교연산자 ] : 비교 결과가 맞으면 True, 아니면 False

    x < y	x가 y보다 작으면 True / 작지 않으면 False
    x > y	x가 y보다 크면 True / 크지 않으면 False
    x == y	x와 y가 같으면 True / 같지 않으면 False
    x != y	x와 y가 같지 않으면 True / 같으면 False
    x >= y	x가 y보다 크거나 같으면 True / 작으면 False
    x <= y	x가 y보다 작거나 같으면 True / 크면 False 
    
'''
# [if 조건문]
close = 100
yes_close = 80

if close > yes_close:
    print("어제보다 종가가 올랐다.")

print("\n*************************************ex2*************************************")

# [if ~ else 조건문]
close = 100
yes_close = 80

if close == yes_close:
    print("같다.")
else:
    print("다르다")

print("\n*************************************ex3*************************************")
# [ if ~ elif ~ else 조건문 ]
close = 100
yes_close = 80

if close > yes_close:
    print("어제보다 종가가 올랐다.")
elif close < yes_close:
    print("어제보다 종가가 떨어졌다.")
else:
    print("어제와 오늘의 종가가 동일하다.")

print("\n*************************************ex4*************************************")

'''
[논리 연산자] : and , or , not 
    x and y:	x와 y 모두 참이어야 참
    x or y: x와 y 둘 중에 하나만 참이어도 참
    not x: x의 반대 값 (x가 참이면 -> 거짓 / 거짓이면 -> 참)
'''
print("True and True:", True and True)
print("True and False:", True and False)
print("False and False:", False and False)

print("True or False:", True or False)
print("True or True:", True or True)
print("False or False:", False or False)

print("not True:", not True)
print("not False:", not False)

print("\n*************************************ex5*************************************")

''' 0, None, 비어있는 값을 제외한 모든 값은 True
    0, None, 비어있는 값은 False로 해석 
       ** None: 아무것도 없다는 것을 나타내는 자료형
'''
print("1 and 0:", 1 and 0)
print("1 and '':", 1 and '')
print("'ABC' and '':", 'ABC' and '')
print("1 and None:", 1 and None)
print("1 or 0:", 1 or 0)
print("1 and 1:", 1 and 1)
print("1 or 1:", 1 or 1)
print("0 and 0:", 0 and 0)
print("0 or 0:", 0 or 0)
print("not 1:", not 1)
print("not 0:", not 0)

print("\n*************************************ex6*************************************")

ex6_a = True
ex6_b = False
print("ex6_a and ex6_b:", ex6_a and ex6_b)

ex6_a2 = 1
ex6_b2 = 0
print("ex6_a2 and ex6_b2:", ex6_a2 and ex6_b2)

ex6_a3 = ''
ex6_b3 = '123'
print("ex6_a3 or ex6_b3:", ex6_a3 or ex6_b3)

ex6_a4 = None
ex6_b4 = '123'
print("ex6_a4 or ex6_b4:", ex6_a3 or ex6_b3)

print("\n*************************************ex7*************************************")
ex7_a = 123
ex7_b = ''
ex7_c = None

if ex7_a:
    print(f"1) ex7_a : {ex7_a}")

if ex7_b:
    print(f"2) ex7_b :{ex7_b}")

if ex7_c:
    print(f"3) ex7_c :{ex7_c}")

if ex7_a or ex7_b:
    print(f"4) ex7_a : {ex7_a}")

if ex7_a and ex7_b:
    print(f"5) ex7_a : {ex7_a}")
