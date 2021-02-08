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

    date = Column(DateTime)
    price = Column(Float)
    volume = Column(Integer)
    market_id = Column(Integer, ForeignKey('market.id'))

    market = relationship("Market", backref=backref('daily_candles', order_by=id))
```
    
Query의 **Join**을 통해 위와 같은 Market과 DailyCandle 테이블의 데이터를 모두 가져올 수 있습니다.   

### join()로 Join

`Query.join()`은 Market과 DailyCandle 사이에 있는 하나의 외래키를 기준으로 join해야 합니다. 

```{.python}
session.query(Market).join(DailyCandle).all()
```



