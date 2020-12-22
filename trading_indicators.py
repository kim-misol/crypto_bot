def bollinger_band(price_df, n=20, k=2):
    bb = price_df.copy()
    bb['mbb'] = price_df['Close'].rolling(n).mean()  # 중앙 이동 평균선
    bb['ubb'] = bb['mbb'] + k * price_df['Close'].rolling(n).std()  # 상단 밴드
    bb['lbb'] = bb['mbb'] - k * price_df['Close'].rolling(n).std()  # 하단 밴드

    return bb

