import os

# print(f"현재 실행 경로: {os.getcwd()}")
# # 현재 실행 경로: C:\Users\Timepercent\Documents\tutorial_coinbot\grammer
# print(f"현재 실행되는 스크립트 파일의 상대 경로: {os.path.dirname(__file__)}")
# # 현재 실행되는 스크립트 파일의 상대 경로: C:/Users/Timepercent/Documents/tutorial_coinbot/grammer
# print(f"현재 실행되는 스크립트 파일의 절대 경로: {os.path.abspath(__file__)}")
# # 현재 실행되는 스크립트 파일의 절대 경로: C:\Users\Timepercent\Documents\tutorial_coinbot\grammer\etc.py
# print(f"현재 실행되는 스크립트 파일이 있는 위치: {os.path.dirname(os.path.abspath(__file__))}")
# # 현재 실행되는 스크립트 파일이 있는 위치: C:\Users\Timepercent\Documents\tutorial_coinbot\grammer

# os.chdir('C:\\Users')
# current_working_directory = os.getcwd()
# print(f"os.chdir(절대경로) 이동 후 경로: {current_working_directory}")
# # C:\Users

# # os.path.relpath(경로1, 경로2) 는 경로2에 대한 경로1의 상대 경로를 출력
# print(os.path.relpath('C:\\Windows\\Fonts', 'C:\\Windows\\System32'))
# print(os.path.relpath('C:\\Windows\\Fonts', 'C:\\Windows'))
# print(os.path.relpath('C:\\Windows\\Fonts', 'C:\\Windows\\Fonts'))
# # 경로2가 없으면 현재 작업 폴더가 기준 경로가 된다.
# print(os.path.relpath('C:\\Windows\\Fonts'))
#
#
# # 특정 경로 내의 모든 파일 및 폴더의 리스트
# print(os.listdir('C:\\Users\\Timepercent\\Documents\\tutorial_coinbot'))
#
# 경로의 유효성 확인
print(os.path.exists('C:\\Windows\\Fonts'))
print(os.path.exists('C:\\abcs!@#@#$'))
# os.path.isdir(경로) 함수는 경로가 존재하며 폴더라면 True를, 경로가 존재하지 않거나 존재하더라도 파일이면 False를 출력
print(os.path.isfile('C:\\Windows\\system32\\notepad.exe'))
# os.path.isfile(경로) 함수는 경로가 존재하며 파일이면 True를, 경로가 존재하지 않거나 존재하더라도 폴더이면 False를 출력
print(os.path.isdir('C:\\Windows\\system32\\notepad.exe'))



