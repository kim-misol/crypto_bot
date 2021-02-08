[TOC]

## 세션 (session)
ORM은 데이터베이스를 session을 이용해 다룰 수 있습니다.  
`create_engine()`과 같은 레벨에서 Session 클래스를 factory 패턴으로 생성할 수 있습니다.

```{.python}
from sqlalchemy.orm import sessionmaker

Session = sessionmaker(bind=engine)
```
Session 클래스는 새 객체를 만들어서 데이터베이스와 연결 됩니다.   
Session 클래스를 생성한 이후부터는 데이터베이스와의 연동이 필요할 때 (Object과 일반적인 데이터베이스 접속) Session을 호출하여 쓰면 됩니다.  
위의 Session은 처음으로 사용될 때 Engine과 연결되고 모든 변경을 커밋하고 세션을 종료할 때까지 열려있게 됩니다.
## 커밋 (commit)
## 롤백 (rollback)
```{.python}
session.rollback()
```
Session이 트랜잭션으로 동작한 후에 rollback을 실행하여 데이터를 변경하기 전 상태로 되돌아 갈 수 있습니다.