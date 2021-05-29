from setting import *

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import torch

import base64
import datetime
import io


def update_epsilon_slider(parameters):
    '''
    update_slider will return a dash component for epsilon sliders 
    this function will be reused for code efficiency

    contains: 1.ws slider 2.wp slider 3.w0 slider 4. epsilon_inf slider
    '''
    epsilon_current_index = parameters["epsilon"]["current_index"]
    selected_w0 = parameters['epsilon']['w0'][epsilon_current_index]
    selected_wp = parameters['epsilon']['wp'][epsilon_current_index]
    selected_ws = parameters['epsilon']['ws'][epsilon_current_index]
    selected_inf = parameters['epsilon']['inf']


    return [
        html.Div(id='epsilon-w0-slider-content',className='slider-content',children=[f'w0: {selected_w0:.2f}']),
        dcc.Slider(
            id='epsilon-w0-slider',
            min=w0_freq_low,
            max=w0_freq_high,
            step=w0_step,
            value=parameters['epsilon']['w0'][epsilon_current_index],
            # updatemode='drag',
            marks={
                w0_freq_low:{"label":f"min:{w0_freq_low}",'style': {'color': 'white'}},
                w0_freq_high:{"label":f"max:{w0_freq_high}",'style': {'color': 'white'}}
            }
            # ,updatemode='drag'
        ),
        
        html.Div(id='epsilon-wp-slider-content',className='slider-content',children=[f'wp: {selected_wp:.2f}']),
        dcc.Slider(
            id='epsilon-wp-slider',
            min=wp_freq_low,
            max=wp_freq_high,
            step=wp_step,
            value=parameters['epsilon']['wp'][epsilon_current_index],
            # updatemode='drag',
            marks={
                wp_freq_low:{"label":f"min:{wp_freq_low}",'style': {'color': 'white'}},
                wp_freq_high:{"label":f"max:{wp_freq_high}",'style': {'color': 'white'}}
            }
        ),
        
        html.Div(id='epsilon-ws-slider-content',className='slider-content',children=[f'ws: {selected_ws:.3f}']),
        dcc.Slider(
            id='epsilon-ws-slider',
            min=ws_freq_low,
            max=ws_freq_high,
            step=ws_step,
            value=parameters['epsilon']['ws'][epsilon_current_index],
            # updatemode='drag',
            marks={
                ws_freq_low:{"label":f"min:{ws_freq_low}",'style': {'color': 'white'}},
                ws_freq_high:{"label":f"max:{ws_freq_high}",'style': {'color': 'white'}}
            }
        ),
        
        html.Div(id='epsilon-inf-slider-content',className='slider-content',children=[f'inf: {selected_inf}']),
        dcc.Slider(
            id='epsilon-inf-slider',
            min=inf_low,
            max=inf_high,
            step=inf_step,
            value=parameters['epsilon']['inf'],
            # updatemode='drag',
            marks={
                inf_low:{"label":f"min:{inf_low}",'style': {'color': 'white'}},
                inf_high:{"label":f"max:{inf_high}",'style': {'color': 'white'}}
            }
        )
    ]

def update_mu_slider(parameters):
    '''
    update_slider will return a dash component for mu sliders 
    this function will be reused for code efficiency
    '''
    mu_current_index = parameters["mu"]["current_index"]
    selected_w0 = parameters['mu']['w0'][mu_current_index]
    selected_wp = parameters['mu']['wp'][mu_current_index]
    selected_ws = parameters['mu']['ws'][mu_current_index]
    selected_inf = parameters['mu']['inf']

    return [
        html.Div(id='mu-w0-slider-content',className='slider-content',children=[f'w_0 : {selected_w0:.2f}'],style={"text-align":"center","font-size":"1.5em"}),
        dcc.Slider(
            id='mu-w0-slider',
            min=w0_freq_low,
            max=w0_freq_high,
            step=w0_step,
            value=parameters['mu']['w0'][mu_current_index],
            # updatemode='drag',
            marks={
                w0_freq_low:{"label":f"min:{w0_freq_low}",'style': {'color': 'white'}},
                w0_freq_high:{"label":f"max:{w0_freq_high}",'style': {'color': 'white'}}
            }
            # ,updatemode='drag'
        ),
        
        html.Div(id='mu-wp-slider-content',className='slider-content',children=[f'w_p : {selected_wp:.2f}'],style={"text-align":"center","font-size":"1.5em"}),
        dcc.Slider(
            id='mu-wp-slider',
            min=wp_freq_low,
            max=wp_freq_high,
            step=wp_step,
            value=parameters['mu']['wp'][mu_current_index],
            # updatemode='drag',
            marks={
                wp_freq_low:{"label":f"min:{wp_freq_low}",'style': {'color': 'white'}},
                wp_freq_high:{"label":f"max:{wp_freq_high}",'style': {'color': 'white'}}
            }
        ),
        
        html.Div(id='mu-ws-slider-content',className='slider-content',children=[f'w_s : {selected_ws:.3f}'],style={"text-align":"center","font-size":"1.5em"}),
        dcc.Slider(
            id='mu-ws-slider',
            min=ws_freq_low,
            max=ws_freq_high,
            step=ws_step,
            value=parameters['mu']['ws'][mu_current_index],
            # updatemode='drag',
            marks={
                ws_freq_low:{"label":f"min:{ws_freq_low}",'style': {'color': 'white'}},
                ws_freq_high:{"label":f"max:{ws_freq_high}",'style': {'color': 'white'}}
            }
        ),
        
        html.Div(id='mu-inf-slider-content',className='slider-content',children=[f'inf: {selected_inf}'],style={"text-align":"center","font-size":"1.5em"}),
        dcc.Slider(
            id='mu-inf-slider',
            min=inf_low,
            max=inf_high,
            step=inf_step,
            value=parameters['mu']['inf'],
            # updatemode='drag',
            marks={
                inf_low:{"label":f"min:{inf_low}",'style': {'color': 'white'}},
                inf_high:{"label":f"max:{inf_high}",'style': {'color': 'white'}}
            }
        )
    ]

reset_component = html.Div('Clear&Reset',id='reset-area')

upload_component = html.Div(id='upload-area',children=[
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag&Drop or ',
            html.A('Select CSV File',style={"text-decoration":"underline","cursor":"pointer"})
        ]),
        style={
            'width': '400px',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '5px',
            'borderStyle': 'solid',
            'borderRadius': '1em',
            'textAlign': 'center',
            'margin': '10px auto',
            'font-size':"1.5em",
            "background":"white"
        }
    ),
    html.Div(id='output-data-upload',style={"text-align":"center"}),
])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    print(content_type)
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    
    return df.to_numpy(),html.Div([
        html.H5(f'Success Upload:{filename}')

        # html.Hr(),  # horizontal line
        # # For debugging, display the raw contents provided by the web browser
        # html.Div('Raw Content'),
        # html.Pre(contents[0:200] + '...', style={
        #     'whiteSpace': 'pre-wrap',
        #     'wordBreak': 'break-all'
        # })
    ])

def init_param(num_lor):
    w0 = np.random.uniform(freq_low, freq_high, num_lor)
    wp = np.random.uniform(wp_freq_low, wp_freq_high, num_lor)
    ws = np.random.uniform(ws_freq_low, ws_freq_high, num_lor)
    eps_inf = 1.
    return w0, wp, ws, eps_inf
def new_param():
    ''' 
        functin for generating one more set of Lorentzian parameters
    '''
    w0 = np.random.uniform(freq_low,freq_high)
    wp = np.random.uniform(0,5)
    ws = np.random.uniform(0,0.05)
    return w0,wp,ws

def reset_parameters(reset_nclicks,old_parameters=None):
    #Initialize Lorentzian parameters randomly for epsilon and mu
    w0,wp,ws,eps_inf=init_param(num_lor_init_epsilon) 
    w0[0]=3
    wp[0]=3
    ws[0]=0.01
    w0m,wpm,wsm,eps_infm = init_param(num_lor_init_mu)

    if old_parameters is  None:
        nclick_epsilon=None
        nclick_mu=None
    else:
        nclick_epsilon =  old_parameters['nclick-epsilon']
        nclick_mu = old_parameters['nclick-mu']

    #Store all related parameters in dict 'parameters'
    parameters={
        "epsilon":{
            "w0":list(w0),
            "ws":list(ws),
            "wp":list(wp),
            "inf":eps_inf,
            "current_index":0
        },
        "mu":{
            "w0":list(w0m), 
            "ws":list(wsm),
            "wp":list(wpm),
            "inf":eps_infm,
            "current_index":0
        },
        "epsilon_num_lor":num_lor_init_epsilon,
        "mu_num_lor":0,
        "nclick-epsilon":nclick_epsilon,
        "nclick-mu":nclick_mu,
        "num_spectra":NUM_SPECTRA,
        "target_spectrum":None,
        "nclick-reset":reset_nclicks
    }

    return parameters