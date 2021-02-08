챕터 별 페이지 내에서 ctrl + f 로 키워드 검색 가능하시면 빠른 검색 가능합니다. 

지속적으로 업데이트 될 예정이며, 모든 질문은 class101 게시판을 이용해 주시기 바랍니다. 

[TOC]


### DDL: Data Definition Language  
**데이터 정의어**로 데이터베이스 혹은 테이블과 같은 데이터 구조를 정의하는 명령어

## CREATE

```{.sql}
CREATE TABLE [IF NOT EXISTS] [table_name] ([create_definition,...]);

CREATE TABLE coin (id BIGINT AUTO_INCREMENT PRIMARY KEY, 
                    name TEXT, 
                    close INT, 
                    volume INT);
```

**coin table**  

| id |   name    |      close      | volume |  
|----|-----------|-----------------|------------|
|  |  |  |  |

## ALTER
```{.sql}
ALTER TABLE [table_name]
    [alter_option [, alter_option] ...]
    [partition_options]
    
ALTER TABLE coin
ADD date TEXT;
```

**coin table**  

| id |   name    |      close      | volume |  date |
|----|-----------|-----------------|------------|---|
|  |  |  |  |  |

## DROP
```{.sql}
DROP TABLE [IF EXISTS]  [table_name];
DROP TABLE IF EXISTS coin;
```

## RENAME
```{.sql}
RENAME TABLE [table_name] TO [new_table_name];
RENAME TABLE coin TO coin_data;
```

**coin_data table**  

| id |   name    |      close      | volume |  
|----|-----------|-----------------|------------|
|  |  |  |  |

## TURNCATE
```{.sql}
TRUNCATE [TABLE] [table_name]
```


