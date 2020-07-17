import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import requests

response = requests.get('https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/RKI_COVID19/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json')

data = []
for i in response.json()['features']:
    for j in i.items():
        data.append(j[1])
rki = pd.DataFrame.from_dict(data, orient='columns')

rki['Meldedatum'] = pd.to_datetime(rki.Meldedatum.astype(int), unit='ms')
rki.set_index('Meldedatum', inplace=True)
rki.sort_index(inplace=True)
rki['AnzahlFall_cumsum'] = rki['AnzahlFall'].cumsum()

myages = ['A00-A04', 'A05-A14', 'A15-A34', 'A35-A59', 'A60-A79', 'A80+']
mylands = ['Sachsen-Anhalt', 'Thüringen', 'Bremen', 'Nordrhein-Westfalen', 'Schleswig-Holstein', 'Hamburg', 'Niedersachsen',
 'Baden-Württemberg', 'Bayern', 'Saarland', 'Berlin', 'Brandenburg',
 'Mecklenburg-Vorpommern', 'Sachsen', 'Hessen', 'Rheinland-Pfalz']

def get_options(list_lands):
    dict_list = []
    for i in list_lands:
        dict_list.append({'label': i, 'value': i})
    return dict_list

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(children=[
                html.Div(className='row',
                    children=[
                        html.Div(className='four columns div-user-controls',
                            children=[
                                html.H2('RKI Covid-19 tracker'),
                                html.P('''Up-to-date Covid-19 data from RKI.'''),
                                html.P('''Pick one or more regions from the dropdown below.'''),  
                                html.Div(className='div-for-dropdown',
                                    children=[
                                    dcc.Dropdown(id='selector',
                                    options=get_options(rki.Bundesland.unique()),
                                    multi=True,
                                    value=[rki['Bundesland'].sort_values()[0]],
                                    style={'backgroundColor': '#1E1E1E'},
                                    className='selector')
                                    ],
                                    style={'color': '#1E1E1E'})    
                                ]),
                        html.Div(className='eight columns div-for-charts bg-grey',
                            children=[
                                dcc.Graph(
                                    id='timeseries', 
                                    config={'displayModeBar': False}),
#                                dcc.Graph(
#                                    id='change', 
#                                    config={'displayModeBar': False})
                            ])
                                ]),
                                ])
    
                                
#dcc.Graph(
#id='timeseries',
#config={'displayModeBar': False},
#animate=True,
#figure=px.line(rki,
#    x=rki.index,
#    y='AnzahlFall_cumsum',
#    color='Bundesland',
#    template='plotly_dark').update_layout(
#    {'plot_bgcolor': 'rgba(0, 0, 0, 0)',
#    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
#    'yaxis_type': 'log'}
#)
#    )                               

@app.callback(Output('timeseries', 'figure'),
              [Input('selector', 'value')])
def update_timeseries(selected_dropdown_value):
    ''' Draw traces of the feature 'value' based one the currently selected Bundesland(s)'''
    # STEP 1
    trace = []  
    df_sub = rki
    # STEP 2
    # Draw and append traces for each stock
    for land in selected_dropdown_value:   
        trace.append(go.Scatter(x=df_sub[df_sub['Bundesland'] == land].index,
                                 y=df_sub[df_sub['Bundesland'] == land]['AnzahlFall_cumsum'],
                                 mode='lines',
                                 opacity=0.7,
                                 name=land,
                                 textposition='bottom center'))  
    # STEP 3
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    # Define Figure
    # STEP 4
    figure = {'data': data,
              'layout': go.Layout(
                  colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
                  title={'text': 'Cumulative cases', 'font': {'color': 'white'}, 'x': 0.5},
                  xaxis={'range': [df_sub.index.min(), df_sub.index.max()]},
              ),
              }
    return figure


if __name__ == '__main__':
    app.run_server(debug=True)
