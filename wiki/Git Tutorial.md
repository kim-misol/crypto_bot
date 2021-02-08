챕터 별 페이지 내에서 ctrl + f 로 키워드 검색 가능하시면 빠른 검색 가능합니다. 

지속적으로 업데이트 될 예정이며, 모든 질문은 class101 게시판을 이용해 주시기 바랍니다. 

[TOC]


### Git
: 소스코드를 효과적으로 관리할 수 있게 해주는 분산 버전 관리 시스템 중 하나입니다. 

### Github, Gitlab
: git으로 저장되어 원격전송된 내역들이 저장되는 공간을 제공하는 서비스 (git hosting platform)  
대체적으로,  
Github은 오픈소스 프로젝트에서 많이 사용되고,  
Gitlab은 기업체 등에서 인트라넷에 설치하여 많이 사용됩니다.

### 저장소 (repository)
: 소스코드가 저장되어 있는 여러 개의 branch가 모여 있는 디스크상의 피지컬 스페이스를 의미합니다.  

저장소는   
1) 로컬 저장소 (local repository)와   
2) 원격 저장소 (remote repository)로 나뉩니다.    

원격 저장소 (remote repository)에서 `clone`을 통해 로컬 저장소 (local repository)로 소스코드를 복제하여 가져오고  
소스코드가 변경되면 `add`, `commit`을 하여 소스코드를 로컬에 저장하고  
`push`하여 원격 저장소 (remote repository)에 반영합니다.  

### 체크아웃 (checkout)
: 특정 branch, commit, tag의 소스코드로 이동하는 것을 의미합니다.

### 스테이지 (stage)
: 작업한 내용이 임시로 올라가는 저장소이며, stage에서 반영할 파일만 선택하여 `commit`합니다.

### 커밋 (commit)
: 작업한 내용을 로컬 저장소 (local repository)에 `commit`하여 저장합니다.

### 태그 (tag)
: commit의 위치를 찾기 쉽도록 붙여 놓는 이정표


### Untracked (비관리대상), Tracked (관리대상)

Untracked file은 파일이 생성된 후 아직 git에 `add` 되지 않은 파일 (track 불가능한 상태)  
Tracked file은 이미 git에 `add`된 적이 있는 파일이며, (staged, modified, unmodified) 중 하나의 상태를 가집니다.  
  


| 관리대상 파일 상태 | 설명 | 
| -------- | -------- | 
| staged     | `add` 된 후 수정되지 않은 상태       |
| modified     | `add` 된 후 수정된 상태      |
| unmodified     | 성공적으로 `commit` 된 상태        |


### .gitignore
: gitignore 파일에 있는 파일/폴더 리스트 무시하기
    



-----




### Git 장점
* 소스 코드를 압축파일로 주고 받을 필요가 없습니다.  
* 프로젝트 진행 시 버전 관리에 용이합니다.  
    *  모든 파일들의 저장된 상태를 원하는 시점으로 되돌릴 수 있습니다.  
    *  이전 버전의 코드와 무엇이 바뀌었는지 확인하기 쉽습니다.




