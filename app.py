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


#Load country ISO codes
code2country = {}
codes = pd.read_csv('data/codes.csv')
code_list = list(codes.Code)
country_list = list(codes.Name)
for i in range(len(code_list)):
    code2country[code_list[i]] = country_list[i]

    
#access API to get up-to-date data
url = "https://thevirustracker.com/timeline/map-data.json"
payload = {}
headers = {}
response = requests.request("GET", 
                            url, 
                            headers=headers, 
                            data=payload)
rawdata = json.loads(response.text.encode('utf8'))
timeline = rawdata['data']
all_df = pd.DataFrame.from_records(timeline)
all_df.index = all_df['date'] = pd.to_datetime(
    all_df['date'], format='%m/%d/%y')
all_df = all_df[['countrycode', 'cases', 
                 'deaths', 'recovered']]
all_df['cases'] = pd.to_numeric(all_df['cases'])
all_df.sort_index(inplace=True)


#calculate daily change in cases 
#and rolling seven day average
country_dfs = []
my_codes = all_df.countrycode.unique()
for code in my_codes:
    if code in code2country:
        _ = all_df[all_df.countrycode == code].copy()
        _.loc[:,'country'] = code2country[code]
        _.loc[:,'prev_cases'] = _.cases.shift(1).fillna(0)
        _.loc[:,'daily_cases_change'] = _.cases - _.prev_cases
        _.loc[:,'daily_cases_change_av'] = _.daily_cases_change.rolling(
            7, min_periods=1).mean(skipna=True)
        country_dfs.append(_)      
final_df = pd.concat(country_dfs)


def predict_country_ar(country, ndays, dataset):
    """Project daily cases numbers for a given country, 
    desired number of days and input dataframe. Currently uses
    an autoregression model from the statsmodels module, and
    predicts based on the rolling seven day average cases."""
    ts_data = dataset[dataset['country'] == country]['daily_cases_change_av']
    ts_data = pd.DataFrame(ts_data)
    batch = ts_data.values
    #makes predictions for successive days,
    #and appends these to the existing data
    #before predicting the next day
    for __ in range(ndays):
        model = AutoReg(batch, lags=1)
        model_fit = model.fit()
        pred = model_fit.predict(len(batch), len(batch))
        batch = np.append(batch, pred)
    preds = batch[-ndays:]
    add_dates = [ts_data.index[-1] +
                 DateOffset(days=x) for x in range(ndays+1)]
    future_dates = pd.DataFrame(index=add_dates[1:], columns=ts_data.columns)
    df_predict = pd.DataFrame(preds,
                              index=future_dates[-ndays:].index, 
                              columns=['prediction'])
    df_proj = pd.concat([ts_data, df_predict], axis=1)
    return df_proj


def get_options(list_countries):
    """Returns a dictionary of label: value
    pairs to use as dropdown options."""
    list_countries = sorted(list_countries)
    dict_list = []
    for i in list_countries:
        dict_list.append({'label': i, 'value': i})
    return dict_list


#run app
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


#define app layout
app.layout = dbc.Container(
    fluid=True, 
    children=[
    dbc.Row(align='center', justify='center', 
            children=
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
                                                   className='selector')
                                  ],
                                  style={'color': '#AAAAAA'}),
                         html.P(''),
                         html.P(
                             '''Weeks ahead to forecast
                             (up to 100 weeks in the future):'''),
                         html.Div(className='dropdown',
                                  children=[
                                      dcc.Input(id='prediction-weeks',
                                                type='number',
                                                value=4, min=1, max=100)
                                  ]),
                         html.P(''),
                         dcc.Graph(
                             id='prediction',
                             config={'displayModeBar': False},
                            style={'height': '41vh'}
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
            width={"size": 4, "offset": 0},
        ),
         dbc.Col(html.Div(className='charts',
                          children=[
                              dcc.Graph(
                                  id='timeseries',
                                  config={
                                      'displayModeBar': False},
                                  style={'height': '47vh'}
                              ),
                              dcc.Graph(
                                  id='change',
                                  config={
                                      'displayModeBar': False},
                                  style={'height': '47vh'}

                              ),
                          ]),
                 width={"size": 6, "offset": 1}
                )
         ], 
        no_gutters=True
         ),
])


#define callbacks to update figures
@app.callback(Output('timeseries', 'figure'),
              [Input('selector', 'value')])
def update_timeseries(selected_dropdown_value):
    ''' Plot log-log growth trajectory 
    curves for the selected countries.'''
    trace = []
    df_sub = final_df
    for country in selected_dropdown_value:
        trace.append(go.Scatter(x=df_sub[df_sub['country'] == country].cases,
                                y=df_sub[df_sub['country'] ==
                                         country].daily_cases_change_av,
                                mode='lines',
                                opacity=0.7,
                                name=country,
                                hovertemplate='%{y:.1s}'))
    data = [val for sublist in [trace] for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(
                      colorway=px.colors.qualitative.Bold,
                      template='plotly_white',
                      margin={'t': 30, 'b': 10},
                      hovermode='x',
                      autosize=True,
                      title={'text': 'Growth trajectory', 
                             'font': {'color': 'black', 'size': 17}, 
                             'x': 0.5, 'y': 0.95},
                      xaxis={'range': [2, math.log(
                          df_sub.cases.max(), 10)], 
                             'type': 'log', 
                             'hoverformat': '.1s',
                             'title': 'Total cases'},
                      yaxis={'type': 'log',
                             'title': 'Change in cases (per day)'}
              )}
    return figure


@app.callback(Output('change', 'figure'),
              [Input('selector', 'value')])
def update_change(selected_dropdown_value):
    ''' Plot daily change in cases for the selected countries.'''
    trace = []
    df_sub = final_df
    for country in selected_dropdown_value:
        trace.append(go.Scatter(x=df_sub[df_sub['country'] == country].index,
                                y=df_sub[df_sub['country'] ==
                                         country].daily_cases_change_av,
                                mode='lines',
                                opacity=0.7,
                                name=country,
                                hovertemplate='%{y:.1s}'))
    data = [val for sublist in [trace] for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(
                      colorway=px.colors.qualitative.Bold,
                      template='plotly_white',
                      margin={'t': 20, 'b': 20},
                      hovermode='x',
                      autosize=True,
                      xaxis={
                          'range': [df_sub.index.min(), df_sub.index.max()],
                      },
                      title={'text': 'Daily change in cases (7 day average)', 
                          'font': {'color': 'black', 'size': 17}, 
                          'x': 0.5, 'y': 0.98},
                      yaxis={'title': 'Change in cases (per day)'}
              )}
    return figure


@app.callback(Output('prediction', 'figure'),
              [Input('selector', 'value'), 
               Input('prediction-weeks', 'value')])
def update_prediction(country_list, weeks):
    '''Plot case forecasts for currently 
    selected countries and 
    desired forecast range.'''
    trace = []
    df_sub = final_df
    #only update graph if forecast 
    #length input field not empty
    if weeks:
        ndays = int(weeks)*7
        for country in country_list:
            df_proj = predict_country_ar(country, ndays, df_sub)
            df_proj = df_proj[-ndays:]
            trace.append(go.Scatter(x=df_proj.index,
                                    y=df_proj.prediction,
                                    mode='markers',
                                    opacity=0.7,
                                    name=country,
                                    hovertemplate='%{y:.1s}'))
        data = [val for sublist in [trace] for val in sublist]
        figure = {'data': data,
                  'layout': go.Layout(
                          colorway=px.colors.qualitative.Bold,
                          template='plotly_white',
                          margin={'t': 40, 'b': 50},
                          hovermode='x',
                          autosize=True,
                          title={'text': 'Autoregression forecast', 
                                 'font': {'color': 'black', 'size': 17}, 
                                 'x': 0.5, 'y': 0.935},
                          yaxis={'title': 'Change in cases (per day)'}
                  )}
        return figure
    else:
        return dash.no_update


if __name__ == '__main__':
    app.run_server(debug=True)
