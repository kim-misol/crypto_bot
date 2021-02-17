import plotly.graph_objects as go


def plot_model_fit_history(code, history, x):
    fig = go.Figure()

    for item in history.history.items():
        y_label = item[0]
        y_value = item[1]
        fig.add_trace(
            go.Scatter(
                x=list(range(x)),
                y=y_value,
                name=y_label
            ))
    fig.update_layout(
        title=code
    )

    return fig


def draw_candle_with_indicator(df, code):
    fig = go.Figure()
    df['Date'] = df.index

    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['ubb'],
            name="볼린저밴드 상한선"
        ))
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['mbb'],
            name="볼린저밴드 중앙선"
        ))
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['lbb'],
            name="볼린저밴드 하한선"
        ))
    fig.add_trace(
        go.Candlestick(x=df['Date'],
                       open=df['open'],
                       high=df['high'],
                       low=df['low'],
                       close=df['close'],
                       increasing_line_color='red', decreasing_line_color='blue',
                       name="캔들"))

    fig.update_layout(title=code, legend_title="Labels")

    return fig


def draw_candle(df, code):
    df['Date'] = df.index

    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'],
                                         increasing_line_color='red', decreasing_line_color='blue')])
    fig.update_layout(
        title=code
    )
    # 특정 시점에 라인 그리기
    # fig.update_layout(
    #     title=code,
    #     shapes=[dict(
    #         x0='2016-12-09', x1='2016-12-09', y0=0, y1=1, xref='x', yref='paper',
    #         line_width=2)],
    #     annotations=[dict(
    #         x='2016-12-09', y=0.05, xref='x', yref='paper',
    #         showarrow=False, xanchor='left', text='Increase Period Begins')]
    # )

    return fig
