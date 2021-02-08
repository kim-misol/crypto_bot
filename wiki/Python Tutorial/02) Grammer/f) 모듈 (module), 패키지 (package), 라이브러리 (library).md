챕터 별 페이지 내에서 ctrl + f 로 키워드 검색 가능하시면 빠른 검색 가능합니다. 

지속적으로 업데이트 될 예정이며, 모든 질문은 class101 게시판을 이용해 주시기 바랍니다. 

[TOC] 
  
  
|  |  |  
| -------- | -------- | 
| module     | 여러 함수와 변수, 클래스를 포함한 Python 파일     | 
| package     | module 묶음     |
| library     | package 묶음     |

`import 모듈명` 또는 `from 모듈명 import 함수, 클래스` 과 같은 방식으로 라이브러리, 패키지 또는 모듈의 [함수](https://wikidocs.net/103533) 또는 [클래스](https://wikidocs.net/103630)를 가져올 수 있습니다.  
`import`는 파일의 최상단에서 하며, 일반적으로 아래의 순서로 import 합니다.  

* Python에 내장된 라이브러리, 패키지, 모듈 (standard library)   
* 설치한 라이브러리, 패키지, 모듈 (third party)   
* 프로젝트 내에 있는 라이브러리, 패키지, 모듈(local application 및 library) 


### import 기본 구조 
```{.python}
import  모듈
from 모듈 import 변수, 함수, 클래스
from 모듈 import 변수 as 이름
from 모듈 import 함수 as 이름
from 모듈 import 클래스 as 이름
```

### import  모듈
예시 1 - math 모듈 임포트하여 sqrt 메서드 사용
```{.python}
import math     # math 라는 모듈에서 sqrt 라는 함수를 선택


num = 4
num_sqrt = math.sqrt(num)
print(f"4의 제곱근: {num_sqrt}")
# 출력: 4의 제곱근: 2.0
```

### from 모듈 import 특정 [함수](https://wikidocs.net/103533)
예시 2 - math 모듈의 sqrt 메서드 임포트하여 사용
```{.python}
from math import sqrt  


num = 4
num_sqrt = sqrt(num)       # sqrt 앞에 math. 을 쓰지 않아도 바로 사용 가능
print(f"4의 제곱근: {num_sqrt}")
# 출력: 4의 제곱근: 2.0
```

### from 모듈 import 모든 [함수](https://wikidocs.net/103533)
예시 3 - math 모듈의 모든 메서드 임포트하여 sql 메서드 사용 **(지양하는 방법)**
math 모듈에 수많은 함수들이 미리 정의가 되어있는데 그 중 모든 함수, 변수, 클래스를 참조 하고 싶을 경우
```{.python}
from math import *


num = 4
num_sqrt = sqrt(num)       # sqrt 앞에 math. 을 쓰지 않아도 바로 사용 가능
print(f"4의 제곱근: {num_sqrt}")
# 출력: 4의 제곱근: 2.0
```

### from 모듈 import [함수](https://wikidocs.net/103533) as 이름
예시 4 - math 모듈의 sqrt 메서드 임포트하여 원하는 이름으로 사용
```{.python}
from math import sqrt as sq


num = 4
num_sqrt = sq(num)
print(f"4의 제곱근: {num_sqrt}")
# 출력: 4의 제곱근: 2.0
```