import pandas as pd
import psycopg2  # db driver


def create_training_engine(db_name):
    conn = psycopg2.connect(dbname=db_name, user="", password="", host="", port="")
    return conn


class DataNotEnough(BaseException):
    pass


def get_table_list():
    engine = create_training_engine()
    sql = f"""SELECT table_name 
FROM information_schema.tables"""
    df = pd.read_sql(sql, engine)
    tables = list(df[:9]['table_name']) + list(df[11:12]['table_name'])
    return tables


def get_kr_market(table_name='market'):
    engine = create_training_engine()
    sql = f"""SELECT * 
    FROM public.{table_name}
    WHERE code like 'KRW-%'"""
    df_kr_market = pd.read_sql(sql, engine)
    return df_kr_market


def get_min_candle(market, unit, table_name="min_candle"):
    yyyy = '2020'
    if unit in (1, 3):
        db_name = 'crypto_bot'
        sql = f"""SELECT *
        FROM public.{table_name}
        WHERE market_id = '{market}'
            AND unit={unit}
            AND created_at_kst >= TO_TIMESTAMP('{yyyy}-01-01 01:00:00', 'YYYY-MM-DD HH:MI:SS')
            ORDER BY created_at_kst DESC;"""
    else:
        # 스키마가 변경되었다
        db_name = 'crypto_bot1'
        sql = f"""SELECT * 
                FROM public.{table_name}
                WHERE market_code = '{market}'
                    AND unit={unit} 
                    AND created_at_kst >= TO_TIMESTAMP('{yyyy}-01-01 01:00:00', 'YYYY-MM-DD HH:MI:SS')
                    ORDER BY created_at_kst DESC;"""
    engine = create_training_engine(db_name)

    mins = pd.read_sql(sql, engine)
    return mins


if __name__ == "__main__":
    # 분봉 1, 3, 5, 10, 15, 30, 60, 240
    unit = int(input(f"unit (분봉)"))
    if unit in (1, 3):
        market = 1
    else:
        # 스키마가 변경되었다
        market = 'KRW-BTC'

    min_candles = get_min_candle(market, unit)



