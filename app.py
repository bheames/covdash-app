import requests
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from pandas.tseries.offsets import DateOffset
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output


code2country = {}
codes = pd.read_csv('data/codes.csv')
code_list = list(codes.Code)
country_list = list(codes.Name)
for i in range(len(code_list)):
    code2country[code_list[i]] = country_list[i]

    
url = "https://thevirustracker.com/timeline/map-data.json"
payload = {}
headers = {}
response = requests.request("GET", url, headers=headers, data=payload)
rawdata = json.loads(response.text.encode('utf8'))
timeline = rawdata['data']
all_df = pd.DataFrame.from_records(timeline)
all_df.index = all_df['date'] = pd.to_datetime(
    all_df['date'], format='%m/%d/%y')
all_df = all_df[['countrycode', 'cases', 'deaths', 'recovered']]
all_df['cases'] = pd.to_numeric(all_df['cases'])
all_df.sort_index(inplace=True)


country_dfs = []
my_codes = all_df.countrycode.unique()
for code in my_codes:
    if code in code2country:
        _ = all_df[all_df.countrycode == code]
        _['country'] = code2country[code]
        _['prev_cases'] = _.cases.shift(1).fillna(0)
        _['daily_cases_change'] = _.cases - _.prev_cases
        _['daily_cases_change_av'] = _.daily_cases_change.rolling(
            7, min_periods=1).mean(skipna=True)
        country_dfs.append(_)      
final_df = pd.concat(country_dfs)


def predict_country_ar(country, dataset, days):
    n_input = days
    ts_data = dataset[dataset['country'] == country]['daily_cases_change_av']
    ts_data = pd.DataFrame(ts_data)
    batch = ts_data.values
    for i in range(n_input):
        model = AutoReg(batch, lags=1)
        model_fit = model.fit()
        pred = model_fit.predict(len(batch), len(batch))
        batch = np.append(batch, pred)
    preds = batch[-n_input:]
    add_dates = [ts_data.index[-1] +
                 DateOffset(days=x) for x in range(n_input+1)]
    future_dates = pd.DataFrame(index=add_dates[1:], columns=ts_data.columns)
    df_predict = pd.DataFrame(preds,
                              index=future_dates[-n_input:].index, columns=['prediction'])
    df_proj = pd.concat([ts_data, df_predict], axis=1)
    return df_proj


def get_options(list_countries):
    list_countries = sorted(list_countries)
    dict_list = []
    for i in list_countries:
        dict_list.append({'label': i, 'value': i})
    return dict_list


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


app.layout = html.Div(children=[
    dbc.Row(
        [dbc.Col(
            html.Div(className='controls',
                     children=[
                         html.P(''),
                         html.H4('COVID-19 tracker'),
                         html.P(
                             '''Pick one or more countries from 
                             the dropdown below to chart growth 
                             trajectories, daily change and forecast
                             future cases.'''),
                         html.Div(className='dropdown',
                                  children=[
                                      dcc.Dropdown(id='selector',
                                                   options=get_options(
                                                       final_df['country'].unique()),
                                                   multi=True,
                                                   value=[
                                                       'United States'],
                                                   style={
                                                       'backgroundColor': '#FFFFFF'},
                                                   className='selector')
                                  ],
                                  style={'color': '#AAAAAA'}),
                         html.P(''),
                         html.P(
                             '''Weeks ahead to forecast.'''),
                         html.Div(className='dropdown',
                                  children=[
                                      dcc.Dropdown(id='prediction-weeks',
                                                   options=get_options(
                                                       range(1,13)),
                                                   multi=False,
                                                   value=2,
                                                   style={
                                                       'backgroundColor': '#FFFFFF'},
                                                   className='selector')
                                  ],
                                  style={'color': '#AAAAAA'}),
                         dcc.Graph(
                             id='prediction',
                             config={'displayModeBar': False},
                         ),
                         html.P(''),
                         html.H6('Data source and code:'),
                         html.P(html.A("https://thevirustracker.com",
                                       href='https://thevirustracker.com',
                                       target="_blank")),
                         html.P(html.A("https://github.com/bheames/covdash-app",
                                       href='https://github.com/bheames/covdash-app',
                                       target="_blank")),
                     ]),
            width={"size": 4, "offset": 1, "justify": "between"}),
         dbc.Col(html.Div(className='charts',
                          children=[
                              dcc.Graph(
                                  id='timeseries',
                                  config={
                                      'displayModeBar': False},
                              ),
                              dcc.Graph(
                                  id='change',
                                  config={
                                      'displayModeBar': False},
                              ),
                          ],
                          style={'textAlign': 'right'}))
         ]),
])


@app.callback(Output('timeseries', 'figure'),
              [Input('selector', 'value')])
def update_timeseries(selected_dropdown_value):
    ''' Draw traces of the feature 'value' based one the currently selected countries'''
    trace = []
    df_sub = final_df
    for country in selected_dropdown_value:
        trace.append(go.Scatter(x=df_sub[df_sub['country'] == country].cases,
                                y=df_sub[df_sub['country'] ==
                                         country].daily_cases_change_av,
                                mode='lines',
                                opacity=0.7,
                                name=country,
                                textposition='bottom center'))
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(
                  colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                            '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_white',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
                  title={'text': 'Growth trajectory', 'font': {'color': 'black'}, 'x': 0.5},
                  xaxis={'range': [2, math.log(
                      df_sub.cases.max(), 10)], 'type': 'log', 'title': 'Total cases'},
                  yaxis={'type': 'log',
                         'title': 'Change in daily cases'}
              ),
              }
    return figure


@app.callback(Output('change', 'figure'),
              [Input('selector', 'value')])
def update_change(selected_dropdown_value):
    ''' Draw traces of the feature 'value' based one the currently selected countries'''
    trace = []
    df_sub = final_df
    for country in selected_dropdown_value:
        trace.append(go.Scatter(x=df_sub[df_sub['country'] == country].index,
                                y=df_sub[df_sub['country'] ==
                                         country].daily_cases_change_av,
                                mode='lines',
                                opacity=0.7,
                                name=country,
                                textposition='bottom center'))
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(
                  colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                            '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_white',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
                  xaxis={
                      'range': [df_sub.index.min(), df_sub.index.max()], 'title': ''},
                  title={'text': 'Daily change in cases (7 day average)', 'font': {'color': 'black'}, 'x': 0.5},
                  yaxis={'title': 'Change in daily change'}
              ),
              }
    return figure


@app.callback(Output('prediction', 'figure'),
              [Input('selector', 'value'), Input('prediction-weeks', 'value')])
def update_prediction(country_list, weeks):
    ''' Draw traces of the feature 'value' based one the currently selected countries'''
    trace = []
    df_sub = final_df
    n_input = weeks*7
    for country in country_list:
        df_proj = predict_country_ar(country, df_sub, n_input)
        df_proj = df_proj[-n_input:]
        trace.append(go.Scatter(x=df_proj.index,
                                y=df_proj.prediction,
                                mode='markers',
                                opacity=0.7,
                                name=country,
                                textposition='bottom center'))
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(
                  colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                            '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_white',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
                  title={'text': 'Autoregression forecast', 'font': {'color': 'black'}, 'x': 0.5},
                  yaxis={'title': 'Predicted change in daily cases'}
              ),
              }
    return figure


if __name__ == '__main__':
    app.run_server(debug=True)
