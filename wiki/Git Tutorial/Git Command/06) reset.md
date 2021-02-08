챕터 별 페이지 내에서 ctrl + f 로 키워드 검색 가능하시면 빠른 검색 가능합니다. 

지속적으로 업데이트 될 예정이며, 모든 질문은 class101 게시판을 이용해 주시기 바랍니다. 

[TOC]

| Git CLI 명령어 | 설명 | 
| -------- | -------- |
| git reset       | local/master의 HEAD가 한단계 뒤로 돌아가며 이후의 commit은 삭제된다. |

main과 branch 양측에서 같은 파일의 같은 라인 코드가 변경된 경우 conflict 나면 reset

# add, commit 한 후에 reset
### 1. git_practice 폴더에 test.txt 파일 생성

![](https://wikidocs.net/images/page/104144/git_reset_add_commit_1.png)

### 2. Sourcetree의 WORKSPACE/파일상태 스테이지에 올라가지 않은 파일 리스트 `test.txt` 파일 

![](https://wikidocs.net/images/page/104144/git_reset_add_commit.png)

### 3. `git add test.txt` 파일 


![](https://wikidocs.net/images/page/104144/git_reset_add_commit_2.png)

### 4. `git commit` 커밋 메세지와 함께 커밋

![](https://wikidocs.net/images/page/104144/git_reset_add_commit_3.png)

### 5. WORKSPACE/History에서 `test.txt` 파일 커밋 내용 확인

![](https://wikidocs.net/images/page/104144/git_reset_add_commit_4.png)

### 6. 되돌리기를 원하는 커밋을 우클릭하여 '커밋 되돌리기' 클릭

![](https://wikidocs.net/images/page/104144/git_reset_add_commit_5.png)

### 7. 사용 중인 모드를 'Hard - 모든 작업 상태 내 변경 사항을 버림' 선택 후 확인 버튼 클릭

![](https://wikidocs.net/images/page/104144/git_reset_add_commit_6.png)

### 8. Revert '되돌아간 커밋의 메세지'라는 커밋 메세지와 함께 커밋 되돌아간 history 확인

![](https://wikidocs.net/images/page/104144/git_reset_add_commit_7.png)

### 9. git_practice 폴더 확인

test.txt 파일 생성 후 커밋한 내용이 초기화 되면서 파일 생성 전 상태로 돌아간 모습

![](https://wikidocs.net/images/page/104144/git_reset_add_commit_8.png)

# 다른 브랜치와의 충돌 발생으로 인한 reset
`pull` 과 `push`를 하는 과정에서 conflict가 발생한다면 아래의 과정 (`reset`)을 통해 로컬의 변경 내용을 conflict 발생 이전으로 되돌릴 수 있습니다. 또한, 되돌린 버전 이후의 버전들은 히스토리에서 삭제 됩니다.

### 1. master branch로 `checkout` 하여 test.txt 파일을 수정

![](https://wikidocs.net/images/page/104144/git_reset_2.png)

### 2. test.txt 파일을 `add`하고 `commit` 한다 (master 브랜치)

![](https://wikidocs.net/images/page/104144/git_reset_3.png)

### 3. master 브랜치 `push`

![](https://wikidocs.net/images/page/104144/git_reset_4.png)

### 4. 브랜치에서 test1 브랜치를 더블 클릭하여 `checkout`

![](https://wikidocs.net/images/page/104144/git_reset_5.png)

### 5. 파일상태 WORKSPACE에서 test.txt 파일을 더블클릭하여 파일을 오픈

![](https://wikidocs.net/images/page/104144/git_reset_6.png)

### 6. test.txt 파일 수정

![](https://wikidocs.net/images/page/104144/git_reset_7.png)

### 7. test.txt 파일을 `add`하고 `commit` 한다 (test1 브랜치)

![](https://wikidocs.net/images/page/104144/git_reset_8.png)

### 8. test1 브랜치에서 master 브랜치를 `pull` 한다

![](https://wikidocs.net/images/page/104144/git_reset_9.png)

### 9. test1과 master 브랜치에서 같은 파일 같은 라인을 수정하여 conflict 발생

![](https://wikidocs.net/images/page/104144/git_reset_10.png)

### 10. test1 브랜치의 History WORKSPACE에서 되돌리고 싶은 버전을 선택
마우스 우클릭하여 '이 커밋까지 현재 브랜치를 초기화' 클릭

![](https://wikidocs.net/images/page/104144/git_reset_20.png)
![](https://wikidocs.net/images/page/104144/git_reset_11.png)

### 11. 사용 중인 모드에서 'Hard - 모든 작업 상태 내 변경 사항을 버림'를 선택하고 '확인'을 클릭

![](https://wikidocs.net/images/page/104144/git_reset_12.png)

### 12. 경고 메세지에 '예' 버튼 클릭
선택한 버전 이후의 모든 버전이 삭제되고, 현재 커밋하지 않은 내용, stage에 올려놓은 내용 전부 삭제된다는 메세지  

![](https://wikidocs.net/images/page/104144/git_reset_13.png)

### 13. 커밋된 내역이 삭제되고 다시 master 브랜치 `pull` 받기



![](https://wikidocs.net/images/page/104144/git_reset_21.png)

![](https://wikidocs.net/images/page/104144/git_reset_22.png)