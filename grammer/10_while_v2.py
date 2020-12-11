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


print("\n*************************************continue, pass, break*************************************")

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

print("\n*************************************else*************************************")

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
