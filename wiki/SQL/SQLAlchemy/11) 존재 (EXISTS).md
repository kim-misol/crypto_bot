```
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

[TOC]

## exists()

SQL에서 `EXISTS` 키워드는 **Boolean** 연산자로 조건에 맞는 행이 있으면 `True`를 반환합니다.  
어떠한 조건을 만족하지 않는 행을 제외하고 데이터를 조회하는 경우에 유용하게 사용할 수 있습니다.

```{.python}
from sqlalchemy.sql import exists

# market 테이블의 is_active이 activate 상태인 market 데이터
stmt = exists().where(Market.is_active=="activate")
for market in session.query(Market).filter(stmt):
    print market
```

  
또한 많은 시나리오에서 관계 테이블의 경우 적합한 값이 없는 행을 처리하는데에도 유용하게 쓰입니다.


```{.python}
from sqlalchemy.sql import exists

# daily_candles와 market 테이블의 데이터 조회
stmt = exists().where(DailyCandle.market_id==Market.id)
for data in session.query(Market).filter(stmt):
    print data
```

## any()
`any()`는 특정 조건에 일치하는 데이터를 제한적으로 필터링 해줍니다.
```{.python}
# market 테이블의 is_active이 activate 상태인 일별 데이터
for daily_candle in session.query(DailyCandle).\
    filter(DailyCandle.market.any(Market.is_active=="activate")):
    print daily_candle
```

## has()

`has()`도 `any()`와 다대일 (One to Many) 관계에서 동일한 기능을 위해 사용됩니다.
```{.python}
# market 테이블의 is_active이 activate 상태인 일별 데이터
session.query(DailyCandle).\
    filter(DailyCandle.market.has(Market.is_active=='activate')).all()
    

# market 테이블의 is_active이 activate 상태가 아닌 일별 데이터 (~연산자:  NOT)
session.query(DailyCandle).\
    filter(~DailyCandle.market.has(Market.is_active=='activate')).all()
```