Query 객체의 `all()` 메서드는 SQL을 호출하여 list 타입의 값을 반환

```{.python}
query = session.query(Coin).filter(Coin.name.like('%코인')).order_by(Coin.date)
query.all()
```

Query 객체의 `one()`, `first()` 메서드는 SQL을 호출하여 non-iterator 타입의 값을 반환
  
`first()`는 첫번째 값을 scalar로 반환 (LIMIT 1)
```{.python}
query = session.query(Coin).filter(Coin.name.like('%코인')).order_by(Coin.date)
query.first()
```

`one()`은 값이 존재하지 않거나 여러 행이 동일한 값을 가지고 있는 경우 에러 발생
```{.python}
from sqlalchemy.orm.exc import NoResultFound, MultipleResultsFound

# 데이터가 존재하지 않은 경우 one()
try:
    user = session.query(Coin).filter(and_(Coin.date=='20210101', Coin.date=='20210102').one()
except NoResultFound, e:
    print e
    
# 동일한 데이터를 가지고 있는 경우 one()
try:
    user = session.query(Coin)).filter(Coin.date.like('2021%')).one()
except MultipleResultsFound, e:
    print e

```