챕터 별 페이지 내에서 ctrl + f 로 키워드 검색 가능하시면 빠른 검색 가능합니다. 

지속적으로 업데이트 될 예정이며, 모든 질문은 class101 게시판을 이용해 주시기 바랍니다. 

[TOC] 

# 절대경로
`절대 경로`란 최초의 시작점으로 경유한 경로를 전부 기입하는 방식입니다.

윈도우 OS의 바탕화면에 위치한 `test.txt `파일이 있다는 가정하에,  
`test.txt`파일의 절대 경로는 최상위 루트 디렉토리인 C 디렉토리 로부터 시작되어 아래와 같이 구성됩니다.
```
C:\Users\UserID\Desktop\test.txt
```

# `pathlib` 모듈
`pathlib` 모듈의 기본 아이디어는 파일시스템 경로르 단순한 문자열이 아닌 객체로 다루자는 것입니다.
```{.python}
from pathlib import Path

path = Path('/user/path/to/file')

print(path)             # \user\path\to\file
print(path.parent)      # \user\path\to
```

# `Path.cwd()`: 현재 경로를 반환
```{.python}
from pathlib import Path

print(Path.cwd())
# 출력: C:\Users\Timepercent\Documents\python\grammer
```
# `Path.exists()`: 해당 경로에 파일이나 디렉터리의 존재 여부
`__file__`은 실행하는 파일의 위치에서부터 파일명까지 표시해줍니다.
```{.python}
from pathlib import Path

path = Path(__file__)
print(path.exists())   # True
```
현재 파일이 존재 여부를 확인하였으므로 `True`를 반환합니다.

`.`는 자신을 `..`는 부모디렉토리를 뜻합니다.
```{.python}
from pathlib import Path

path = Path('.')
p = path / Path('현재 파일이 위치한 디렉토리의 부모 디렉토리에 존재하는 파일.py')
p.exists()             # False

parent_path = Path('..')
p1 = parent_path / Path('현재 파일이 위치한 디렉토리의 부모 디렉토리에 존재하는 파일.py')
p1.exists()            # True
```

