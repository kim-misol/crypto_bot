ORM에서는 데이터베이스 테이블을 사용할 수 있게 설정한 다음 정의한 클래스에 맵핑을 해야 합니다.  
sqlalchemy에서는 두가지가 동시에 이뤄지는데 1) **Declarative** 를 이용하여 클래스를 생성하고 2) 실제 데이터베이스 테이블에 연결을 합니다.  
  
```{.python}
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
```  
  
위와 같이 declarative_base() 함수를 import 하면 여러개의 매핑 클래스를 만들 수 있습니다.   
매핑 클래스 내에서 모델을 생성하기 위해 테이블의 컬럼을 나타내는 Column 클래스, 각 컬럼의 데이터타입을 나타내는 Integer, String, DateTime 등의 클래스를 불러와야 합니다.
    
   
```{.python}
class Coin(Base):
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime(timezone=True))
    code = Column(String(50), index=True)
    name = Column(String(50))
    price = Column(Integer)
    volume = Column(Integer)
```

MySQL 데이터베이스의 경우 데이터타입을 명시할 때 varchar 컬럼 길이 length가 필요하여 `Column(String(50))` 과 같이 설정해 주어야 합니다.
Integer, Numeric 같은 경우에도 위와 동일하게 쓸 수 있습니다.

## 새로운 객체 추가


```{.python}
new_coin = Coin('20200101', 'BTC/KRW', '비트코인', 8300000, 103)
session.add(new_coin)
```

`session.add()` 후에도 실제로 데이터베이스에는 추가되기 전인 **pending**인 상태입니다.   
아직 데이터베이스에 발행되지는 않은 상태인데 입력이 필요한 순간에는 flush라는 과정을 통해 입력이 됩니다.

```{.python}
session.add_all([
    Coin('20200101', 'BTC/KRW', '비트코인', 8300000, 103),
    Coin('20200102', 'BTC/KRW', '비트코인', 8037000, 205),
    Coin('20200103', 'BTC/KRW', '비트코인', 8474000, 589)
```
`session.add_all()` 으로 한번에 여러 데이터를 추가할 수도 있습니다.

```{.python}
session.commit()
```

`commit()`은 **pending**되어 있던 추가, 변경된 이력을 데이터베이스에 반영합니다.