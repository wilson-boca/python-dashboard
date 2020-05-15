import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import datetime
import quandl
import requests
import pandas as pd
import json
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np

from dash.dependencies import Input, Output, State
from dateutil.relativedelta import relativedelta

quandl.ApiConfig.api_key = "29yfsH7sPvDMZV7e8rjx"
start = datetime.datetime.today() - relativedelta(years=5)
end = datetime.datetime.today()

df = quandl.get('EOD/MSFT', start_date=start, end_date=end, returns="pandas")
layout = dict(title="EOD Microsoft", showlegend=False, xaxis=dict(rangeslider_visible=True))
trace_close = go.Scatter(x=list(df.index), y=list(df.Close), name="Value", line=dict(color='#ff0000'))
data1 = [trace_close]
data2 = [trace_close]
fig1 = px.line(data1, x=list(df.index), y=list(df.Close), title='Séries com Rangeslider', labels={'x': 'Mês/Ano', 'y': 'Valor de Fechamento'})
fig1.update_xaxes(rangeslider_visible=True)
fig2 = dict(data=data2, layout=layout)


def update_news():
    url = 'https://www.quandl.com/api/v3/datasets/WIKI/FB.json?column_index=4&start_date=2017-01-01&end_date=2017-12-31&collapse=monthly&transform=rdiff&api_key=29yfsH7sPvDMZV7e8rjx'
    result = requests.get(url)
    json_result = json.loads(result.text)
    df_data = pd.DataFrame(json_result['dataset']['data'])
    return df_data


def generate_html_table(max_rows=10):
    return html.Div(
        [
            html.Div(
                html.Table(
                    [html.Tr([html.Th('Data'), html.Th('Valor de Fechamento')])]
                    +
                    [
                        html.Tr(
                            [
                                html.Td(df_data.iloc[i][0]),
                                html.Td(df_data.iloc[i][1])
                            ]
                        )
                        for i in range(min(len(df_data), max_rows))
                    ]
                ),
                style={"height": "300px", "overflowY": "scroll"},
            ),
        ],
        style={"height": "100%"}, )


app = dash.Dash(__name__)

df_data = update_news()
data_canada = px.data.gapminder().query("country == 'Canada'")
fig_bar = px.bar(data_canada, x='year', y='pop')

# Add histogram data
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2
x4 = np.random.randn(200) + 4
hist_data = [x1, x2, x3, x4]
group_labels = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)

labels = ['Oxygen', 'Hydrogen', 'Carbon_Dioxide', 'Nitrogen']
values = [4500, 2500, 1053, 500]
piechart = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0, 0.2, 0])])

us_cities = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv")
map = px.scatter_mapbox(us_cities, lat="lat", lon="lon", hover_name="City", hover_data=["State", "Population"],
                        color_discrete_sequence=["fuchsia"], zoom=5, height=300)
map.update_layout(mapbox_style="open-street-map")
map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})


app.layout = html.Div([
    html.Div([
        html.H1(children="Python Dash/Plotly"),
        html.Img(src="/assets/stock-icon-white.png")
    ], className='banner'),
    html.Div([
        html.Div([
            dcc.Dropdown(id='stock-drop',
                         options=[{'label': 'Walt Disney', 'value': 'EOD/DIS'},
                                  {'label': '3M', 'value': 'EOD/MMM'},
                                  {'label': 'Nike', 'value': 'EOD/NKE'},
                                  {'label': 'Intel', 'value': 'EOD/INTC'}],
                         value='EOD/DIS'),
            html.Div(dcc.Graph(id="stock-chart1", figure=fig1), )
        ], className='six columns'),
        html.Div([
            html.Div([
                dcc.Input(id='stock-input',
                          placeholder='Digite a ação',
                          type='text',
                          value='EOD/INTC'),
                html.Button(id='submit-stock', n_clicks=0, children='enviar')
            ]),
            html.Div(dcc.Graph(id="stock-chart2", figure=fig2))
        ], className='six columns'),
        html.Div([
            html.Div([
                html.H3(children='Ações do Facebook no ano de 2017'),
                html.Div(generate_html_table()),
            ], className='six columns'),
            html.Div([
                html.H3(children='População no Canadá'),
                html.Div(dcc.Graph(figure=fig_bar)),
            ], className='six columns')
        ]),
        html.Div([
            html.Div([
                html.H3(children='Pizzas'),
                html.Div(dcc.Graph(figure=piechart)),
            ], className='six columns'),
            html.Div([
                html.H3(children='População no Canadá'),
                html.Div(dcc.Graph(figure=fig)),
            ], className='six columns')
        ]),
        html.Div(dcc.Graph(id="map", figure=map))

    ])
])


@app.callback(dash.dependencies.Output('stock-chart1', 'figure'),
              [dash.dependencies.Input('stock-drop', 'value')])
def update_stock(input_value):
    if input_value is None:
        return px.line([], x=[], y=[], title='Empty data, please select an option...')
    df = quandl.get(input_value, start_date=start, end_date=end, returns="pandas")
    fig1 = px.line(data1, x=df.index, y=df.Close, title='Séries com Rangeslider', labels={'x': 'Mês Ano', 'y': 'Valor de Fechamento'})
    fig1.update_xaxes(rangeslider_visible=True,
                      rangeselector=dict(
                          buttons=list([
                              dict(count=1, label="1 Mês", step="month", stepmode="backward"),
                              dict(count=6, label="6 Meses", step="month", stepmode="backward"),
                              dict(count=1, label="1 Ano", step="year", stepmode="backward"),
                              dict(step="all", label="Tudo")
                          ])
                      )
    )
    return fig1


@app.callback(dash.dependencies.Output('stock-chart2', 'figure'),
              [dash.dependencies.Input('submit-stock', 'n_clicks')],
              [State("stock-input", 'value')])
def update_stock(n_clicks, input_value):
    df = quandl.get(input_value, start_date=start, end_date=end, returns="pandas")
    trace_line = go.Scatter(x=df.index,
                            y=df.Close,
                            visible=True,
                            name="Close",
                            showlegend=False)

    trace_candle = go.Candlestick(x=df.index,
                                  open=df.Open,
                                  high=df.High,
                                  low=df.Low,
                                  visible=False,
                                  close=df.Close,)
    data = [trace_line, trace_candle]
    update_menus = list([
        dict(
            buttons=list([
                dict(
                    args=[{'visible': [True, False]}],
                    label='Line',
                    method='update'
                ),
                dict(
                    args=[{'visible': [False, True]}],
                    label='CandleStick',
                    method='update'
                )
            ]),
            direction='down',
            pad={'r': 10, 't': 10},
            showactive=True,
            x=0,
            xanchor='left',
            y=1.05,
            yanchor='top'
        ),
    ])

    layout = dict(title='{}'.format(input_value),
                  showlegend=False,
                  updatemenus=update_menus)
    return {
        "data": data,
        "layout": layout
    }


if __name__ == "__main__":
    app.run_server(debug=True)
