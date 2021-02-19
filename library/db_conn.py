import datetime

import numpy as np
import pandas as pd
from sqlalchemy.event import listen
import psycopg2  # db driver
from sqlalchemy.exc import InternalError, ProgrammingError


def create_training_engine():
    conn = psycopg2.connect(dbname="crypto_bot", user="postgres", password="1234", host="192.168.0.118", port="5432")
    return conn


class DataNotEnough(BaseException):
    pass


def get_table_list():
    engine = create_training_engine()
    sql = f"""SELECT table_name 
FROM information_schema.tables"""
    df = pd.read_sql(sql, engine)
    tables = list(df[:9]['table_name']) + list(df[11:12][
                                                   'table_name'])  # ['market', 'min_candle', 'day_candle', 'week_candle', 'month_candle', 'ticker', 'min_indicator', 'day_indicator', 'week_indicator', 'month_indicator']
    return tables

    # with engine:
    #     with engine.cursor() as cursor:
    #         sql = f"""SELECT table_name
    #         FROM information_schema.tables"""
    #         cursor.execute(sql)
    #         results = cursor.fetchall()


def get_kr_market(table_name='market'):
    engine = create_training_engine()
    sql = f"""SELECT * 
    FROM public.{table_name}
    WHERE code like 'KRW-%'"""
    df_kr_market = pd.read_sql(sql, engine)
    return df_kr_market


def get_min_candle(market_id, table_name="min_candle"):
    engine = create_training_engine()
    sql = f"""SELECT * 
    FROM public.{table_name}
    WHERE market_id = '{market_id}'
        AND unit=1 LIMIT 1000000"""
    mins = pd.read_sql(sql, engine)
    return mins


if __name__ == "__main__":
    get_table_list()
    market_list = get_kr_market()
    for market_id in market_list['id']:
        min_candles = get_min_candle(market_id=market_id)


