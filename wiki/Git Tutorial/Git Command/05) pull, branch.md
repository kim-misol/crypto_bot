챕터 별 페이지 내에서 ctrl + f 로 키워드 검색 가능하시면 빠른 검색 가능합니다. 

지속적으로 업데이트 될 예정이며, 모든 질문은 class101 게시판을 이용해 주시기 바랍니다. 

[TOC]

| Git CLI 명령어 | 설명 | 
| -------- | -------- |
| git pull        | 원격 저장소의 데이터를 모두 가져와서 자동으로 merge한다. |
| git branch      | 현재 존재하는 branch를 조회한다. -r 옵션을 사용하면 원격저장소의 브랜치를 확인 가능 |
| git branch 브랜치명 | 새로운 브랜치를 만듭니다. |
| git checkout 브랜치명/태그명    | 해당 branch나 tag로 작업트리를 변경 |


![](https://wikidocs.net/images/page/104143/draw_git_pull.png)  
  
# Remote Repository의 소스코드 pull

### Gitlab의 'git_practice' repository에서 새로운 파일 생성  

![](https://wikidocs.net/images/page/104143/git_pull_ENvLDoU.png)

### 파일명, 파일 내용, 커밋 메세지 입력 후 [Commit changes] 버튼 클릭

![](https://wikidocs.net/images/page/104143/git_pull_2_9gseNIi.png)

### 생성된 파일 확인

![](https://wikidocs.net/images/page/104143/git_pull_3_dEilqqv.png)

### Sourcetree에서 Pull 버튼 클릭
Pull 우측 상단의 [1]은 pull 받을 내용이 1개 있다는 알림입니다.

![](https://wikidocs.net/images/page/104143/git_pull_4.png)

### 원격 저장소, 원격 브랜치, 옵션을 확인 후 Pull 버튼 클릭

![](https://wikidocs.net/images/page/104143/git_pull_5.png)

### Pull 받아진 내용 History에서 확인

![](https://wikidocs.net/images/page/104143/git_pull_6_6zw2BSK.png)

### git_practice 폴더 확인
`test.txt` 파일이 git_practice 폴더에 생성된 것을 확인하실 수 있습니다.
![](https://wikidocs.net/images/page/104143/git_pull_7.png)
# branch

## branch 생성

![](https://wikidocs.net/images/page/104143/git_branch.png)

![](https://wikidocs.net/images/page/104143/git_branch_2.png)

![](https://wikidocs.net/images/page/104143/git_branch_3.png)
## checkout (branch 변경)
**test1** 브랜치에서 **mater** 브랜치로 이동할 경우 sourcetree 의 브랜치 리스트에서 **master** 를 더블클릭 해줍니다.   

![](https://wikidocs.net/images/page/104143/git_checkout.png)  
  
**master** 브랜치로 변경된 모습  

![](https://wikidocs.net/images/page/104143/git_checkout_2.png)  

# 다른 branch의 변경사항 pull

## 소스코드 수정

**mater** 브랜치에서 `test.txt` 파일 수정 후 **test1** 브랜치에서 변경사항을 pull 받아보겠습니다.

파일 상태가 '전체'를 선택 후 `test.txt` 파일을 더블클릭하여 메모장을 엽니다.  

![](https://wikidocs.net/images/page/104143/%ED%8C%8C%EC%9D%BC_%EC%88%98%EC%A0%95.png)  

아래 예시와 같이 수정 후 `Ctrl + S` 단축키로 파일을 저장한 후 닫아줍니다.  

![](https://wikidocs.net/images/page/104143/%ED%8C%8C%EC%9D%BC_%EC%88%98%EC%A0%95_2.png)

파일 수정 후 Souretree에 변경된 모습  

![](https://wikidocs.net/images/page/104143/%ED%8C%8C%EC%9D%BC_%EC%88%98%EC%A0%95_3.png)

## add, commit

`git add` 와 `git commit` 을 하여 **로컬 저장소** (local repository)에 저장  

[add 하는 방법](https://wikidocs.net/104133#add)  
[commit 하는 방법](https://wikidocs.net/104133#commit)  

![](https://wikidocs.net/images/page/104143/%ED%8C%8C%EC%9D%BC_%EC%88%98%EC%A0%95_4.png)

## push

`git push origin master`: **master** 브랜치에서 `push` 하여 **원격 저장소** (remote repository)에 변경 사항 적용

![](https://wikidocs.net/images/page/104143/git_push_from_mater.png)

![](https://wikidocs.net/images/page/104143/git_push_from_mater_2.png)

## checkout 
**mater** 브랜치에서 **test1** 브랜치로 이동

## pull
**test1** 브랜치에서 `test.txt` 파일을 보면 변경사항이 적용되지 않는 것을 볼 수 있습니다.

![](https://wikidocs.net/images/page/104143/git_pull.png)  
  
가져오기 위한 원격 브랜치를 **master** 로 설정한 후 **pull** 버튼 클릭  

![](https://wikidocs.net/images/page/104143/git_pull_2.png)    

**pull** 받아서 변경된 사항 적용된 결과

![](https://wikidocs.net/images/page/104143/git_pull_3.png)


