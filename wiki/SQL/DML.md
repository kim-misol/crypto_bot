챕터 별 페이지 내에서 ctrl + f 로 키워드 검색 가능하시면 빠른 검색 가능합니다. 

지속적으로 업데이트 될 예정이며, 모든 질문은 class101 게시판을 이용해 주시기 바랍니다. 

[TOC]


### DML: Data Manipulation Language  
**데이터 조작어**로 데이터베이스에 데이터를 저장, 읽기, 업데이트, 삭제하는 명령어  

아래와 같은 users 테이블이 존재한다는 가정하에   

| id |   name    |      close      | volume |  
|----|-----------|-----------------|------------|
|  1 | A | 1000 | 230000 |
|  2 | B | 1200 | 340000 |
  
## Create  
```{.sql}
INSERT INTO [table_name] ([columns]) VALUES ([values]);
INSERT INTO users (name, close, volume) VALUES ("B", 1200, 340000);
```

## Read  
```{.sql}
SELECT * FROM [table_name];
SELECT * FROM users;
```  

-- output  

| id |   name    |      close      | volume |  
|----|-----------|-----------------|------------|
|  1 | A | 1000 | 230000 |
|  2 | B | 1200 | 340000 |
  
```{.sql}
SELECT * FROM [table_name] WHERE [condition];
SELECT * FROM users WHERE name = 'B';
```  

-- output  

| id |   name    |      close      | volume |  
|----|-----------|-----------------|------------|
|  2 | B | 1200 | 340000 |

  
## Update  
```{.sql}
UPDATE [table_name] SET [columns] = [values];
UPDATE users SET volume = 350000;
```  

-- before changed   

| id |   name    |      close      | volume |  
|----|-----------|-----------------|------------|
|  1 | A | 1000 | 230000 |
|  2 | B | 1200 | 340000 |

-- output  

| id |   name    |      close      | volume |  
|----|-----------|-----------------|------------|
|  1 | A | 1000 | 350000 |
|  2 | B | 1200 | 350000 |


```{.sql}
UPDATE [table_name] SET [columns] = [values] WHERE [condition];
UPDATE users SET volume = 350000 WHERE name = 'A';
```

-- before changed  

| id |   name    |      close      | volume |  
|----|-----------|-----------------|------------|
|  1 | A | 1000 | 230000 |
|  2 | B | 1200 | 340000 |

-- output  

| id |   name    |      close      | volume |  
|----|-----------|-----------------|------------|
|  1 | A | 1000 | 350000 |
|  2 | B | 1200 | 340000 |

## Delete  
```{.sql}
DELETE FROM [table_name];
DELETE FROM users;
```

-- before changed

| id |   name    |      close      | volume |  
|----|-----------|-----------------|------------|
|  1 | A | 1000 | 230000 |
|  2 | B | 1200 | 340000 |

-- output

| id |   name    |      close      | volume |  
|----|-----------|-----------------|------------|
|  |  |  |  |

```{.sql}
DELETE FROM [table_name] WHERE [condition];
DELETE FROM users WHERE name = 'A';
```

-- before changed

| id |   name    |      close      | volume |  
|----|-----------|-----------------|------------|
|  1 | A | 1000 | 230000 |
|  2 | B | 1200 | 340000 |

-- output

| id |   name    |      close      | volume |  
|----|-----------|-----------------|------------|
|  2 | B | 1200 | 340000 |