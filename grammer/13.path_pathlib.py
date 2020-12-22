from pathlib import Path

"""
pathlib 모듈의 기본 아이디어는 파일시스템 경로르 단순한 문자열이 아닌 객체로 다루자는 것입니다.
가령 폴더 또는 파일 존재성 여부를 아래와 같이 판단할 수 있습니다.
"""
path = Path('/user/path/to/file')

print(path)
print(path.parent)


print("\n************************************* 실행한 스크립트 폴더경로 얻기 *************************************")
cwd = Path.cwd()
print(cwd)

# print("\n************************************* 이 디렉터리 트리에 있는 소스파일 나열 *************************************")
# path = Path('.')
# files = list(path.glob('*'))
# print(files)

print("\n************************************* 디렉터리 트리 내에서 탐색 *************************************")
print(__file__)
path = Path(__file__)
print(path.exists())

print("\n************************************* 파일 유효성 *************************************")
path = Path('.')
p = path / Path('README.md')
print(p.exists())          # False

parent_path = Path('..')
p1 = parent_path / Path('README.md')
print(p1.exists())          # True