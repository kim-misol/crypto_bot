챕터 별 페이지 내에서 ctrl + f 로 키워드 검색 가능하시면 빠른 검색 가능합니다. 

지속적으로 업데이트 될 예정이며, 모든 질문은 class101 게시판을 이용해 주시기 바랍니다. 

[TOC]

# About PyCharm
 **PyCharm**은 JetBrains에 의해 개발된 Python 개발용 IDE (통합 개발 환경)입니다.
 
IDE는 Compile, Code Editor, Debugger, Deploy 등의 모든 작업을 하나의 프로그램에서 할 수 있도록 해주는 통합 개발 환경을 제공합니다. 

# PyCharm 단축키


| 기능 | 단축키 |
| -------- | -------- |
| 프로젝트 내 전체 파일을 검색     | **Ctrl + Shift + F**     | 
| 현재 파일 내에서 검색 |  **Ctrl + F** |
| 프로젝트 내 전체 파일을 대체 | **Ctrl + Shift + R** |
| 현재 파일 내에서 대체 | **Ctrl + R** |
| 라인 자동 정렬 | **Ctrl + Alt + L** |
| 변수 또는 함수 정의된 라인으로 이동 | **Ctrl + (클릭)** |
| 커서 이전 위치로 이동 | **Ctrl + Alt + (왼쪽 방향키)** |
| Project Settings 열기 | **Ctrl + Alt + S** |  

# Project Interpreter 

1. **File** -> **Settings** -> **Project Interpreter** 클릭 또는 단축키 **Ctrl + Alt + S**를 실행하여 Settings 윈도우창을 열어 project interpreter `<no interpreter>` 부분을 클릭  
![](https://wikidocs.net/images/page/102711/pycharm_interpreter.png)  

1. **Show All...**를 클릭  
![](https://wikidocs.net/images/page/102711/pycharm_interpreter_2.png)  

1. **[+]** 버튼을 클릭  
![](https://wikidocs.net/images/page/102711/pycharm_interpreter_3.png)  

1. 모든 창 **[OK]** 버튼을 클릭   
![](https://wikidocs.net/images/page/102711/pycharm_interpreter_4.png)  


# Python Package 설치 (CLI)
Python에서 필요한 패키지를 설치하기 위해 **pip**를 많이 사용합니다.  
`pip install [패키지명]`을 입력하면 원하는 패키지를 설치할 수 있습니다.  

1. 하단의 Terminal 탭을 클릭
![](https://wikidocs.net/images/page/102711/pip_install_package.png)

1. `pip install [패키지명]`을 입력하여 패키지를 설치 (예시, pandas)
![](https://wikidocs.net/images/page/102711/pip_install_package_2.png)

# Python Package 설치 (GUI)


Pycharm에서는 패키지 쉽게 설치할 수 있도록 도와주는 기능이 있습니다. 

![](https://wikidocs.net/images/page/102711/pip_install.png)  

위에 캡쳐 화면과 같이  **File** -> **Settings** -> **Project Interpreter** ->**+ 버튼**을 클릭하면  
아래와 같이 패키지를 고를 수 있는 창이 뜹니다.

![](https://wikidocs.net/images/page/102711/pip_install_2.png)  

여기서 원하시는 패키지를 선택한 후 ‘install Package’ 버튼을 클릭하면 해당 패키지가 설치됩니다.

가끔 pip install 명령어로 설치한 패키지가 잘 실행되지 않는 경우, 파이참의 해당 메뉴를 이용해서 설치하면 정상 작동하는 경우도 있습니다.

# Coding assistance 

Python 코드가 많아지고, 함수나 클래스가 여러 개가 되면 구조를 파악하기가 어려워집니다.  
PyCharm에는 파이썬 파일 및 함수, 변수, 클래스 등의 구조를 쉽게 파악할 수 있게 도와주는 기능이 있어 구조를 파악하기에 용이합니다.   
    
편집창 좌측에는 Project 탭과 Structure탭이 있습니다.  
Project 탭은 해당 폴더 안에 있는 파이썬 파일들을 리스트로 보여줍니다. 폴더 또는 파일 선택 및 편집도 가능합니다.   
Structure탭은 현재 코드 안에 있는 함수들을 모두 보여줍니다. 더블클릭하면 해당 함수로 이동도 가능합니다. 코드가 길어져 작성한 함수를 찾기 힘들 때 굉장히  유용한 기능입니다.  
  
이 외에도 pycharm에서는 python을 쉽게 코딩하기 위한 다양한 방법들을 제공해줍니다. [디버깅](https://wikidocs.net/102711#debug)을 위한 기능도 있고, Pandas 데이터 프레임을 엑셀처럼 보여주는 기능도 있습니다.


# Debug
디버그 (Debug) 또는 디버깅 (Debugging)은 프로그램 개발 도중 발생하는 논리적 오류 또는 비정상적인 연산을 찾아내고 수정하는 과정을 뜻합니다.  
PyCharm의 디버거 (Debugger)는 원하는 코드에 중단지점 (Break Point)를 쉽게 정하여 브레이크 포인트가 걸린 코드를 실행하기 전에 정지하고, 메모리에 저장되어 있는 값을 출력해보거나 코드를 단계적으로 실행하며 분석하는 것을 돕는 도구입니다.  

원하는 코드에 브레이크 포인트를 찍은 후   
**현재 열려있는 창이 지정되어 있는지 확인 후 디버그 버튼 클릭**  
![](https://wikidocs.net/images/page/102711/debuggibg.png)    

**현재 열려있는 창이 지정되어 있지 않다면 열려있는 파일을 우클릭하여 디버그 실행**  
![](https://wikidocs.net/images/page/102711/debuggibg.png)    
      
# Ctrl + D (소스코드 비교 기능)

PyCharm에는 아래의 사진과 같이 Ctrl를 누른 상태에서 코드를 비교할 파일을 선택한 후  **Ctrl D** 단축키를 누르면  
  
![](https://wikidocs.net/images/page/102711/compare_code.png)  
  
코드가 다른 부분 (변경되거나 삭제된 코드)이 하이라이트 되어 쉽게 비교 할 수 있습니다.  
  
![](https://wikidocs.net/images/page/102711/compare_code_2.png)

# Database Plugin 추가


1. **Ctrl Alt S 단축키로 Settings 윈도우를 연 뒤 왼쪽에서 Plugins 선택**  
  ![](https://wikidocs.net/images/page/102711/add_db_plugin.png)  

2. **Marketplace에서 'Database Navigator' 검색 후 [Install] 버튼 클릭  **  
  ![](https://wikidocs.net/images/page/102711/add_db_plugin_2.png)  

3. **[Restart IDE] 버튼 클릭 후 [Restart] 클릭**
  ![](https://wikidocs.net/images/page/102711/add_db_plugin_3.png)  
  
  
4. **View > Tool Windows > DB Browser**
  ![](https://wikidocs.net/images/page/102711/add_db_plugin_4.png)  
  


5. **[+] 버튼 클릭 후 MySQL 클릭  **  
  ![](https://wikidocs.net/images/page/102711/add_db_plugin_5.png)  
  

6. **Database 정보 기입 후 [OK] 클릭  **
  ![](https://wikidocs.net/images/page/102711/add_db_plugin_6.png)  
  
7. **Console 창 더블클릭으로 열거나 'New SQL Console'으로 새로 생성**  
  Datagrip 또는 MySQL Workbench와 동일하게 Ctrl Enter를 통해 쿼리를 실행할 수 있습니다.  
  ![](https://wikidocs.net/images/page/102711/add_db_plugin_7.png)

-----


### Database Plugin 추가 시 유의사항
#### The server time zone value is unrecognized or represents more than one time zone 에러 발생하는 경우
**SET Statement 실행**
```{.sql}
mysql> SELECT @@GLOBAL.time_zone, @@SESSION.time_zone;
+--------------------+---------------------+
| @@GLOBAL.time_zone | @@SESSION.time_zone |
+--------------------+---------------------+
| SYSTEM             | SYSTEM              |
+--------------------+---------------------+
1 row in set (0.00 sec)

mysql> SET GLOBAL time_zone = '+9:00';
Query OK, 0 rows affected (0.01 sec)

mysql> SET time_zone = '+9:00';
Query OK, 0 rows affected (0.00 sec)

mysql> SET GLOBAL time_zone = '+9:00';
Query OK, 0 rows affected (0.00 sec)

mysql> SELECT @@GLOBAL.time_zone, @@SESSION.time_zone;
+--------------------+---------------------+
| @@GLOBAL.time_zone | @@SESSION.time_zone |
+--------------------+---------------------+
| +09:00             | +09:00              |
+--------------------+---------------------+
1 row in set (0.00 sec)
```
  
