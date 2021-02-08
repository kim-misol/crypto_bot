Query 객체의  `count()` 메서드는 조건에 맞는 행의 수를 반환합니다.

```{.python}
session.query(Coin).filter(Coin.date=='20210101').count()
```
위 코드로 생성되는 SQL을 살펴보면, SQLAlchemy는 항상 어떤 쿼리가 오더라도 거기서 행의 수를 셉니다.   
`SELECT count(*) FROM coin` 하면 단순해지지만 최근 버전의 SQLAlchemy는 정확한 SQL로 명시적으로 판단할 수 있는 경우 추측해서 처리하지 않습니다.  
  
숫자를 세야 할 필요가 있는 경우에는 아래와 같이 `func.count()`로 명시적으로 작성하면 됩니다.

```{.python}
from sqlalchemy import func

# coin 테이블의 데이터 수
session.query(func.count('*')).select_from(Coin).scalar()
# coin 테이블의 날짜별 데이터 수
session.query(func.count(Coin.date), Coin.date).group_by(Coin.date).all()
```

User의 primary key를 사용하면 select_from 없이 사용할 수 있습니다.

```{.python}
# coin 테이블의 데이터 수
session.query(func.count(Coin.id)).scalar()
```
