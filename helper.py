from setting import *

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

def update_epsilon_slider(parameters):
    '''
    update_slider will return a dash component for epsilon sliders 
    this function will be reused for code efficiency
    '''
    epsilon_current_index = parameters["epsilon"]["current_index"]

    return html.Div(id='epsilon-slider-area',className='slider',children=[
        
        html.Div(id='epsilon-w0-slider-content',className='slider-content',children=[],style={"text-align":"center","font-size":"1.5em"}),
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
        
        html.Div(id='epsilon-wp-slider-content',className='slider-content',children=[],style={"text-align":"center","font-size":"1.5em"}),
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
        
        html.Div(id='epsilon-ws-slider-content',className='slider-content',children=[],style={"text-align":"center","font-size":"1.5em"}),
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
        
        html.Div(id='epsilon-inf-slider-content',className='slider-content',style={"text-align":"center","font-size":"1.5em"}),
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
    ])

def update_mu_slider(parameters):
    '''
    update_slider will return a dash component for mu sliders 
    this function will be reused for code efficiency
    '''
    mu_current_index = parameters["mu"]["current_index"]

    return html.Div(id='mu-slider-area',className='slider',children=[
        
        html.Div(id='mu-w0-slider-content',className='slider-content',children=[],style={"text-align":"center","font-size":"1.5em"}),
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
        
        html.Div(id='mu-wp-slider-content',className='slider-content',children=[],style={"text-align":"center","font-size":"1.5em"}),
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
        
        html.Div(id='mu-ws-slider-content',className='slider-content',children=[],style={"text-align":"center","font-size":"1.5em"}),
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
        
        html.Div(id='mu-inf-slider-content',className='slider-content',style={"text-align":"center","font-size":"1.5em"}),
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
    ])