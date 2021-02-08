챕터 별 페이지 내에서 ctrl + f 로 키워드 검색 가능하시면 빠른 검색 가능합니다. 

지속적으로 업데이트 될 예정이며, 모든 질문은 class101 게시판을 이용해 주시기 바랍니다. 

[TOC]

| Git CLI 명령어 | 설명 | 
| -------- | -------- |
| [git add](https://wikidocs.net/104133)         | 파일을 새로 추척하거나 수정된 파일을 staged로 반영 |
| [git commit](https://wikidocs.net/104133) -m "[메세지]"  | git add로 추가된 파일을 메세지와 함께 commit |



![](https://wikidocs.net/images/page/104133/draw_git_add_commit.png)
# add
PyCharm에서 git_practice 디렉토리의 `test.txt` 파일을 생성  

![](https://wikidocs.net/images/page/104133/git_add.png)  

파일 생성되면 스테이지에 올라가지 않은 파일 목록에 `test.txt` 파일이 자동 추가  
파일 옆의 **[+] 버튼**을 클릭할 경우, 파일을 하나씩 `add` 할 수 있고  
**[스테이지에 올라가지 않는 파일]** 을 클릭할 경우, 새로 생성되거나 수정된 파일을 한번에 모두 `add` 할 수 있습니다.

![](https://wikidocs.net/images/page/104133/git_add_2.png)  


# commit

### sourcetree에서 commit
하단의 텍스트 입력란에 커밋 메세지를 입력한 후 **[커밋]** 버튼을 클릭하면 로컬 저장소에 커밋됩니다.  
  
![](https://wikidocs.net/images/page/104133/git_commit.png)  

### 로컬 저장소에 반영
`commit`은 로컬 저장소에만 반영되므로 원격 저장소 (remote repository)에는 변경 사항이 적용되지 않습니다.  


**원격 저장소**에 변경 사항을 적용하기 위해서는 `commit` 후 `push`를 해주어야 합니다.
  
![](https://wikidocs.net/images/page/104133/gitlab_repository.png)

