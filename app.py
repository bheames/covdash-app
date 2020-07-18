import requests
import pandas as pd
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

def get_options(list_countries):
    dict_list = []
    for i in list_countries:
        dict_list.append({'label': i, 'value': i})
    return dict_list

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div(children=[
#                html.Div(className='row',
                dbc.Row(
#                    children=[
                    [dbc.Col(
                        html.Div(className='four columns div-user-controls',
                            children=[
                                html.H2('Covid-19 tracker'),
                                html.P('Data source:'),
                                html.A("https://thevirustracker.com", href='https://thevirustracker.com', target="_blank"),
                                html.P(''),
                                html.P('''Pick one or more countries from the dropdown below:'''),  
                                html.Div(className='div-for-dropdown',
                                    children=[
                                    dcc.Dropdown(id='selector',
                                    options=get_options(final_df['country'].unique()),
                                    multi=True,
#                                    value=[final_df['country'].sort_values()[0]],
                                    value=['United Kingdom'],
                                    style={'backgroundColor': '#FFFFFF'},
                                    className='selector')
                                    ],
                                    style={'color': '#AAAAAA'})    
                                ]), width={"size": 3, "offset": 1, "justify": "between"}),
                        dbc.Col(html.Div(className='eight columns div-for-charts bg-grey',
                            children=[
                                    dcc.Graph(
                                    id='timeseries', 
                                    config={'displayModeBar': False},
                                    style={'height': '50vh'}),
                                    dcc.Graph(
                                    id='change', 
                                    config={'displayModeBar': False},
                                    style={'height': '50vh'})
                            ],
                                style={'textAlign': 'right'}))
                                ]),
                                ])

@app.callback(Output('timeseries', 'figure'),
              [Input('selector', 'value')])
def update_timeseries(selected_dropdown_value):
    ''' Draw traces of the feature 'value' based one the currently selected countries'''
    # STEP 1
    trace = []  
    df_sub = final_df
    # STEP 2
    # Draw and append traces for each stock
    for country in selected_dropdown_value:   
        trace.append(go.Scatter(x=df_sub[df_sub['country'] == country].cases,
                                y=df_sub[df_sub['country'] == country].daily_cases_change_av,
                                mode='lines',
                                opacity=0.7,
                                name=country,
                                textposition='bottom center'))  
    # STEP 3
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    # Define Figure
    # STEP 4
    figure = {'data': data,
              'layout': go.Layout(
                  colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_white',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
                  title={'text': 'Growth trajectory', 'font': {'color': 'black'}, 'x': 0.5},
                  xaxis={'range': [2, math.log(df_sub.cases.max(), 10)], 'type': 'log'},
                  yaxis={'type': 'log'}
              ),
              }
    return figure


@app.callback(Output('change', 'figure'),
              [Input('selector', 'value')])
def update_timeseries(selected_dropdown_value):
    ''' Draw traces of the feature 'value' based one the currently selected countries'''
    # STEP 1
    trace = []  
    df_sub = final_df
    # STEP 2
    # Draw and append traces for each stock
    for country in selected_dropdown_value:   
        trace.append(go.Scatter(x=df_sub[df_sub['country'] == country].index,
                                y=df_sub[df_sub['country'] == country].daily_cases_change,
                                mode='lines',
                                opacity=0.7,
                                name=country,
                                textposition='bottom center'))  
    # STEP 3
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    # Define Figure
    # STEP 4
    figure = {'data': data,
              'layout': go.Layout(
                  colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_white',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
                  title={'text': 'Daily change', 'font': {'color': 'black'}, 'x': 0.5},
                  xaxis={'range': [df_sub.index.min(), df_sub.index.max()],},
              ),
              }
    return figure

#dcc.Graph(
#id='timeseries',
#config={'displayModeBar': False},
#animate=True,
#figure=px.line(final_df,
#    x='cases',
#    y='daily_cases_change_av',
#    color='country',
#    template='plotly_dark').update_layout(
#    {'plot_bgcolor': 'rgba(0, 0, 0, 0)',
#    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
#    'yaxis_type': 'log',
#    'xaxis_type': 'log',
#    'xaxis_range': ('1.5','7')})
#    )

if __name__ == '__main__':
    app.run_server(debug=True)