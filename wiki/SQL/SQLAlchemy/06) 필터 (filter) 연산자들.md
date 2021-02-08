[TOC]

### equals
```{.python}
# 날짜가 2021년 1월 1일인 데이터
coins = session.query(Coin).filter(Coin.date == '20210101')
```

### not equals
```{.python}
# 날짜가 2021년 1월 1일 아닌 데이터
coins = session.query(Coin).filter(Coin.date != '20210101')
```

### LIKE

```{.python}
# 날짜가 2021년인 데이터
coins = session.query(Coin).filter(Coin.date.like('2021%'))
    
# 날짜가 1월 1일인 데이터
coins = session.query(Coin).filter(Coin.date.like('%0101'))
    
# 이름에 '코인'을 포함한 데이터
coins = session.query(Coin).filter(Coin.name.like('%코인%'))
```

### IN

```{.python}
# 날짜가 2021년 1월 1~3일의 데이터
coins = session.query(Coin).filter(Coin.date.in_(['20210101', '20210102', '20210103']))

# 서브쿼리도 가능
# Daily_Buy 테이블의 2021년 1월 1일에 거래된 코인의 데이터
coins = session.query(Coin).filter(Coin.name.in_(session.query(Daily_Buy.name).filter(Daily_Buy.date=='20210101')))
```

### NOT IN
```{.python}
날짜가 2021년 1월 1~3일인 데이터를 제외한 데이터
coins = session.query(Coin).filter(~Coin.date.in_(['20210101', '20210102', '20210103']))
```

### IS NULL

```{.python}
# 가격 데이터가 비어있는 데이터
coins = session.query(Coin).filter(Coin.date == None)
```
filter(User.name == None)

### IS NOT NULL

```{.python}
# 가격 데이터가 비어있지 않은 데이터
coins = session.query(Coin).filter(Coin.price != None)
```

### AND

```{.python}
from sqlalchemy import and_

# 날짜가 2021년 1월 1일이면서 코인 이름이 '비트코인'인 데이터
coins = session.query(Coin).filter(and_(Coin.date == '20210101', Coin.name == '비트코인'))

# 다른 방법
coins = session.query(Coin).filter(Coin.date == '20210101').filter(Coin.name == '비트코인')
```

### OR

```{.python}
from sqlalchemy import or_

# 날짜가 2021년 1월 1일이거나  2021년 1월 2일인 데이터
coins = session.query(Coin).filter(or_(Coin.date == '20210101', Coin.date == '20210102'))
```

### 문자열 SQL

`filter()`의 파라미터에서 콜론(**:**)을 이용과 문자열 기반의 SQL를 사용할 수 있습니다.  
Dynamic Value를 사용할 때 `param()` 메서드를 통해 값을 전달합니다.

```{.python}
session.query(Coin).filter("price>:price and date=:date").\
    params(price=10000, date='20210101').order_by(Coin.price).all()
```

`from_statement()`를 통해 문자열 기반의 일반적인 쿼리를 사용할 수 있습니다. 컬럼들은 클래스에서 선언된 것과 동일하게 써야합니다.

```{.python}
session.query(Coin).from_statement(
                    "SELECT * FROM coin WHERE name=:name").\
                    params(name='비트코인').all()
                    
session.query("name", "price", "date").\
        from_statement(
                    """SELECT name, price, date 
                    FROM coin 
                    WHERE name=:name""").\
                    params(name='비트코인').all()
```