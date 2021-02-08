Query 객체는 `session`에서 `query()` method로 생성됩니다.   
이 함수는 다양한 수의 argument를 가질 수 있는데 다양한 클래스의 조합과 클래스 descriptor를 사용할 수 있습니다.   

```{.python}
for instance in session.query(Coin).order_by(Coin.date):
    print(instance.name, instance.price)
    
for row in session.query(Coin, Coin.name).all():
    print row.Coin, row.name
```
Query는 KeyedTuple 클래스 통해 tuple 형태로 반환하는데 일반적인 python 객체처럼 활용할 수 있습니다.   
각 저장된 값들은 클래스 이름이나 속성 이름과 동일합니다.  


-----


LIMIT이나 OFFSET을 포함한 기본적인 Query 동작은 order by와 함께 파이썬 배열에서 슬라이싱(slicing) 방식을 이용하면 됩니다.


```{.python}
for coin in session.query(Coin).order_by(Coin.id)[1:3]:
    print coin
```


-----

필터링이 필요한 경우 `filter_by()` 또는 `filter()`를 이용하면 됩니다.
`filter()` 를 쓰면 클래스 단위의 속성과 파이썬 표준 연산자와 같이 `filter_by()` 보다 더 유연한 SQL 표현을 쓸 수 있습니다.

```{.python}
for date, name in session.query(Coin.date, Coin.name).filter_by(date='20210101'):
    print (date, name)
    
  for date, name in session.query(Coin.date, Coin.name).filter(Coin.date=='20210101'):
    print name
```
