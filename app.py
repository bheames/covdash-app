from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
import requests
import pandas as pd
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
headers= {}
response = requests.request("GET", url, headers=headers, data = payload)
rawdata = json.loads(response.text.encode('utf8'))
timeline = rawdata['data']
all_df = pd.DataFrame.from_records(timeline)
all_df.index = all_df['date'] = pd.to_datetime(all_df['date'], format='%m/%d/%y')
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
        _['daily_cases_change_av'] = _.daily_cases_change.rolling(7, min_periods=1).mean(skipna=True)
        country_dfs.append(_)

final_df = pd.concat(country_dfs)

prediction_dict = {}

n_input = 20

def predict_country(my_country, dataset):
    ts_data = dataset[dataset['country'] == my_country]['daily_cases_change_av']
    ts_data = pd.DataFrame(ts_data)
    train = ts_data

    n_input = 20
    n_features = 1
    
    scaler = MinMaxScaler()
    scaler.fit(train)
    train = scaler.transform(train)

    generator = TimeseriesGenerator(train, train, length=n_input, batch_size=30)

    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dropout(0.25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(generator, epochs=100, verbose=0)

    pred_list = []

    batch = train[-n_input:].reshape((1, n_input, n_features))

    for i in range(n_input):
        pred_list.append(model.predict(batch)[0])
        batch = np.append(batch[:, 1:,:], [[pred_list[i]]], axis=1)

    add_dates = [ts_data.index[-1] + DateOffset(days = x) for x in range(21)]

    future_dates = pd.DataFrame(index=add_dates[1:], columns=ts_data.columns)

    df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                             index=future_dates[-n_input:].index, columns=['prediction'])

    df_proj = pd.concat([ts_data, df_predict], axis=1)
    df_proj['country'] = my_country
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
                                html.P('''Pick one or more countries from the dropdown below to chart growth trajectories and daily change [right]:'''),  
                                html.Div(className='dropdown',
                                    children=[
                                    dcc.Dropdown(id='selector',
                                    options=get_options(final_df['country'].unique()),
                                    multi=True,
#                                    value=[final_df['country'].sort_values()[0]],
                                    value=['United Kingdom'],
                                    style={'backgroundColor': '#FFFFFF'},
                                    className='selector')
                                    ],
                                    style={'color': '#AAAAAA'}),
                                html.P(''),
                                html.P('''Additionally, pick one or more countries for LSTM-based forecasting of daily cases [below]. Allow 10-20 seconds for successive predictions.'''),
                                html.Div(className='dropdown',
                                    children=[
                                    dcc.Dropdown(id='prediction-selector',
                                    options=get_options(final_df['country'].unique()),
                                    multi=True,
                                    value=['United Kingdom'],
                                    style={'backgroundColor': '#FFFFFF'},
                                    className='selector')
                                    ],
                                    style={'color': '#AAAAAA'}),
                                dcc.Graph(
                                    id='prediction', 
                                    config={'displayModeBar': False},
                                    style={'height': '50vh'}),
                                html.H6('Data source and code:'),
                                html.P(html.A("https://thevirustracker.com", 
                                       href='https://thevirustracker.com', 
                                       target="_blank")),
                                html.P(html.A("https://github.com/bheames/covdash-app", 
                                       href='https://github.com/bheames/covdash-app', 
                                       target="_blank")),
                                ]), width={"size": 4, "offset": 1, "justify": "between"}),
                        dbc.Col(html.Div(className='charts',
                            children=[
                                    dcc.Graph(
                                    id='timeseries', 
                                    config={'displayModeBar': False},
                                    style={'height': '50vh'}),
                                    dcc.Graph(
                                    id='change', 
                                    config={'displayModeBar': False},
                                    style={'height': '50vh'}
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
                                y=df_sub[df_sub['country'] == country].daily_cases_change_av,
                                mode='lines',
                                opacity=0.7,
                                name=country,
                                textposition='bottom center'))  
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(
                  colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_white',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
#                  title={'text': 'Growth trajectory', 'font': {'color': 'black'}, 'x': 0.5},
                  xaxis={'range': [2, math.log(df_sub.cases.max(), 10)], 'type': 'log', 'title': 'Total cases'},
                  yaxis={'type': 'log', 'title': 'Change in daily cases (7 day average)'}
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
                                y=df_sub[df_sub['country'] == country].daily_cases_change_av,
                                mode='lines',
                                opacity=0.7,
                                name=country,
                                textposition='bottom center'))  
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(
                  colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_white',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
#                  title={'text': 'Daily change in total case numbers', 'font': {'color': 'black'}, 'x': 0.5},
                  xaxis={'range': [df_sub.index.min(), df_sub.index.max()], 'title': ''},
                  yaxis={'title': 'Change in daily change (7 day average)'}
              ),
              }
    return figure

@app.callback(Output('prediction', 'figure'),
              [Input('prediction-selector', 'value')])
def update_prediction(selected_dropdown_value):
    ''' Draw traces of the feature 'value' based one the currently selected countries'''
    trace = []  
    df_sub = final_df
    global prediction_dict
    for country in selected_dropdown_value: 
        if country in prediction_dict:
            df_proj = prediction_dict[country]
        else:
            df_proj = predict_country(country, df_sub)
            df_proj = df_proj[-n_input:]
            prediction_dict[country] = df_proj
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
                  colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_white',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
#                  title={'text': 'Growth trajectory', 'font': {'color': 'black'}, 'x': 0.5},
#                  xaxis={'title': 'Date'},
                  yaxis={'title': 'Predicted daily cases'}
              ),
              }
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
