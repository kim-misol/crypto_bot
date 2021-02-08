[TOC]

Market 테이블과 DailyCandle 테이블을 **One to Many Relationship**을 만드는 경우,
Market 테이블의 daily_candles와 DailyCandle의 market를 통해 연결시켜주는 예제입니다.

```{.python}
class Market(Base):
    __tablename__ = 'market'

    id = Column(Integer, primary_key=True)
    code = Column(String(20), nullable=False, index=True, unique=True)
    name = Column(String(150), index=True)
    is_active = Column(Boolean, index=True)
    daily_candles = relationship("DailyCandle", backref="market")
    
class DailyCandle(Base, CandleBase):
    __tablename__ = 'daily_candle'

    date = Column(DateTime(timezone=True))
    price = Column(Float)
    volume = Column(Integer)
    market_id = Column(Integer, ForeignKey('market.id'))
    
    market = relationship("Market", backref=backref('daily_candles', order_by=id))
```


 위에서 작성한 `ForeignKey`는 `daily_candle.market_id` 컬럼이 `market.id` 컬럼을 따르도록 만드는 역할입니다.  
`relationship()`은 ORM에게 **DailyCandle** 클래스 자체가 **Market** 클래스에 연결되어 있다는 사실을 `DailyCandle.market` 속성을 이용해 알 수 있게 해줍니다.  
  
 `relationship()`내에서 호출하는 `backref()`는 역으로 클래스를 이용할 수 있도록, 즉 **DailyCandle** 객체에서 **Market**를 참조할 수 있도록 `Market.daily_candles` 구현합니다.   
   
 `DailyCandle.market`와 `Market.daily_candles` 관계는 양방향 관계(bidirectional relationship)로 SQLAlchemy ORM의 주요 특징입니다.
 
 
## Reationship에서의 연산자

### __eq__() 
 
```{.python}
# 다대일(Many to One)에서의 equals 비교
query.filter(DailyCandle.market == some_market)
```

### __ne__() 
 
```{.python}
# 다대일(Many to One)에서의 not equals 비교
query.filter(DailyCandle.market != some_market)
```

### IS NULL 
 
```{.python}
# 다대일(Many to One) 비교 (__eq__())
query.filter(DailyCandle.market == None)
```

### contains() 
 
```{.python}
# 일대다(One to Many) 컬렉션에서 사용
query.filter(Market.daily_candles.contains(some_daily_candles))
```

### any() 
 
```{.python}
# 컬렉션에서 사용
query.filter(Market.daily_candles.any(DailyCandle.date == '20210101'))

# 키워드 아규먼트도 받음
query.filter(Market.daily_candles.any(date='20210101'))
```

### has() 
```{.python}
# scalar 레퍼런스서 사용
query.filter(DailyCandle.market.has(is_active='activate'))
```

### Query.with_parent() 어떤 관계서든 사용

```{.python}
session.query(DailyCandle).with_parent(some_market, 'daily_candles')
```