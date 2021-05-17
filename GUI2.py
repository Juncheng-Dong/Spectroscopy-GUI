import torch

import numpy as np
import pandas as pd
from plotly import __version__
import plotly.offline as pyo
import plotly.graph_objects as go
import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

num_lor=2

freq_low=1
freq_high=5

def init_param():
    w0 = torch.tensor(np.random.uniform(freq_low, freq_high, num_lor), requires_grad=True)
    wp = torch.tensor(np.random.uniform(0, 5, num_lor), requires_grad=True)
    ws = torch.tensor(np.random.uniform(0, 0.05, num_lor), requires_grad=True)
    eps_inf = torch.tensor(10., requires_grad=True)
    d = torch.tensor(50., requires_grad=True)
    return w0, wp, ws, eps_inf, d

def new_param():
    ''' 
        functin for generating one more set of Lorentzian parameters
    '''
    w0 = np.random.uniform(freq_low,freq_high)
    wp = np.random.uniform(0,5)
    ws = np.random.uniform(0,0.05)

    return w0,wp,ws

#Initialize Lorentzian parameters randomly for epsilon and mu
w0,wp,ws,eps_inf,d=init_param() 
w0m,wpm,wsm,eps_infm,dm = init_param()

#Store all related parameters in dict 'parameters'
parameters={
    "epsilon":{
        "w0":list(w0),
        "ws":list(ws),
        "wp":list(wp),
        "inf":eps_inf.item(),
        "current_index":0
    },
    "mu":{
        "w0":list(w0m),
        "ws":list(wsm),
        "wp":list(wpm),
        "inf":eps_infm.item(),
        "current_index":0
    },
    "epsilon_num_lor":num_lor,
    "mu_num_lor":num_lor,
    "nclick-epsilon":None,
    "nclick-mu":None
}
#generate equally separated points from 1THz to 5THz
w=np.linspace(freq_low,freq_high,2001)

#Initialize the App
app=dash.Dash(__name__)
#region# ################### Epsilon Related Section ##########################
epsilon_current_index = parameters["epsilon"]["current_index"]


epsilon_slider_area_left = html.Div(id='epsilon-slider-area',className='slider',children=[
        dcc.Slider(
            id='epsilon-w0-slider',
            min=1,
            max=5,
            step=0.01,
            value=parameters['epsilon']['w0'][epsilon_current_index],
            # updatemode='drag',
            marks={
                1:{"label":"min:1",'style': {'color': 'white'}},
                5:{"label":"max:5",'style': {'color': 'white'}}
            }
        ),
        html.Div(id='epsilon-w0-slider-content',children=[],style={"text-align":"center","font-size":"1.5em"}),

        dcc.Slider(
            id='epsilon-wp-slider',
            min=0,
            max=5,
            step=0.01,
            value=parameters['epsilon']['wp'][epsilon_current_index],
            # updatemode='drag',
            marks={
                0:{"label":"min:0",'style': {'color': 'white'}},
                5:{"label":"max:5",'style': {'color': 'white'}}
            }
        ),
        html.Div(id='epsilon-wp-slider-content',children=[],style={"text-align":"center","font-size":"1.5em"}),
        dcc.Slider(
            id='epsilon-ws-slider',
            min=0,
            max=0.05,
            step=0.001,
            value=parameters['epsilon']['ws'][epsilon_current_index],
            # updatemode='drag',
            marks={
                0:{"label":"min:0",'style': {'color': 'white'}},
                0.05:{"label":"max:0.05",'style': {'color': 'white'}}
            }
        ),
        html.Div(id='epsilon-ws-slider-content',children=[],style={"text-align":"center","font-size":"1.5em"})
    ],style={"width":"50%","float":"left"})

epsilon_slider_area_right=html.Div(className='slider',children=[
    dcc.Slider(
        id='epsilon-inf-slider',
        min=0,
        max=100,
        step=1,
        value=parameters['epsilon']['inf'],
        # updatemode='drag',
        marks={
            0:{"label":"min:0",'style': {'color': 'white'}},
            100:{"label":"max:100",'style': {'color': 'white'}}
        }
    ),
    html.Div(id='epsilon-inf-slider-content',style={"text-align":"center","font-size":"1.5em"}),

    dcc.Dropdown(
        id='epsilon-index-dd',
        options=[{'label': k+1, 'value': k} for k in range(num_lor)],
        value=0
    ),

    html.Button("Add 1 more Lorentzian",id='epsilon-add-button',className='button',style={"display":"inline-block","font-size":"1.5em","margin":"1em","width":"90%","padding":"0.5em"})
    
],style={"width":"50%","float":"right"})

@app.callback(
    Output(component_id='epsilon-slider-area',component_property='children'),
    Input(component_id='epsilon-index-dd',component_property='value')
)
def epsilon_update_index(selected_index):
    current_index=selected_index
    parameters['epsilon']['current_index']=current_index
    return html.Div([dcc.Slider(
            id='epsilon-w0-slider',
            min=1,
            max=5,
            step=0.01,
            value=parameters['epsilon']['w0'][current_index],
            # updatemode='drag',
            marks={
                1:{"label":"min:1",'style': {'color': 'white'}},
                5:{"label":"max:5",'style': {'color': 'white'}}
            }
        ),
        html.Div(id='epsilon-w0-slider-content',children=[],style={"text-align":"center","font-size":"1.5em"}),

        dcc.Slider(
            id='epsilon-wp-slider',
            min=0,
            max=5,
            step=0.01,
            value=parameters['epsilon']['wp'][current_index],
            # updatemode='drag',
            marks={
                0:{"label":"min:0",'style': {'color': 'white'}},
                5:{"label":"max:5",'style': {'color': 'white'}}
            }
        ),
        html.Div(id='epsilon-wp-slider-content',children=[],style={"text-align":"center","font-size":"1.5em"}),
        dcc.Slider(
            id='epsilon-ws-slider',
            min=0,
            max=0.05,
            step=0.001,
            value=parameters['epsilon']['ws'][current_index],
            # updatemode='drag',
            marks={
                0:{"label":"min:0",'style': {'color': 'white'}},
                0.05:{"label":"max:0.05",'style': {'color': 'white'}}
            }
        ),
        html.Div(id='epsilon-ws-slider-content',children=[],style={"text-align":"center","font-size":"1.5em"})
    ])

@app.callback(
    [Output(component_id='epsilon-main-graph',component_property='figure'),
    Output(component_id='epsilon-w0-slider-content',component_property='children'),
    Output(component_id='epsilon-wp-slider-content',component_property='children'),
    Output(component_id='epsilon-ws-slider-content',component_property='children'),
    Output(component_id='epsilon-inf-slider-content',component_property='children'),
    Output(component_id='epsilon-index-dd',component_property='options')],
    
    [Input(component_id='epsilon-w0-slider',component_property='value'),
    Input(component_id='epsilon-wp-slider',component_property='value'),
    Input(component_id='epsilon-ws-slider',component_property='value'),
    Input(component_id='epsilon-inf-slider',component_property='value'),
    Input('epsilon-add-button', 'n_clicks')]
)
def epsilon_update_graph(selected_w0,selected_wp,selected_ws,selected_inf,nclick):

    if nclick==None:
        pass

    if nclick != parameters['nclick-epsilon']:
        parameters['nclick-epsilon'] = nclick
        parameters['epsilon_num_lor'] = parameters['epsilon_num_lor']+1
        # parameters['epsilon']['current_index'] = parameters['num_lor']+1

        #adding parameters
        w0_new, wp_new,ws_new =  new_param()
        parameters['epsilon']['w0'].append(w0_new)
        parameters['epsilon']['wp'].append(wp_new)
        parameters['epsilon']['ws'].append(ws_new)
    
    current_index = parameters['epsilon']['current_index']
    parameters['epsilon']['w0'][current_index]=selected_w0
    parameters['epsilon']['wp'][current_index]=selected_wp
    parameters['epsilon']['ws'][current_index]=selected_ws
    parameters['epsilon']['inf']=selected_inf


    content1 = f'w_0: {selected_w0:.2f}'
    content2 = f'w_p: {selected_wp:.2f}'
    content3 = f'w_s: {selected_ws:.5f}'
    content4 = f'inf: {selected_inf}'


    w0 = torch.tensor(parameters['epsilon']['w0'])
    wp = torch.tensor(parameters['epsilon']['wp'])
    ws = torch.tensor(parameters['epsilon']['ws'])
    eps_inf = torch.tensor(parameters['epsilon']['inf'])

    num_spectra=2001
    num_lor = parameters['epsilon_num_lor']

    w0 = w0.unsqueeze(1).expand(num_lor, num_spectra)
    wp = wp.unsqueeze(1).expand_as(w0)
    ws = ws.unsqueeze(1).expand_as(w0)
    w_expand = torch.tensor(w).expand_as(ws)

    num = pow(wp,2)
    denum = pow(w0,2)-pow(w,2)+(1j)*ws*w
    epi_r = eps_inf + torch.sum(torch.div(num,denum),axis=0)

    fig= go.Figure()
    fig.add_trace(go.Scatter(x=list(w),y=list(epi_r.real),name='real part'))
    fig.add_trace(go.Scatter(x=list(w),y=list(epi_r.imag),name='imaginary part'))
    fig.add_trace(go.Scatter(x=[parameters['epsilon']['w0'][current_index]],y=[0],name='current peak',marker_size=16))
    fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
    ),title=r"$\epsilon_r$", title_x=0.5,yaxis_range=[-200,200],
    margin=dict(l=20, r=20, t=40, b=20))
    
    return fig, content1, content2, content3, content4, [{'label': k+1, 'value': k} for k in range(num_lor)]


@app.callback(Output(component_id='epsilon-button-content',component_property='children'), [Input('epsilon-add-button', 'n_clicks')])
def epsilon_on_click(nclick):
    return html.Div(children=[f'total number of Lorentzians: ',html.Span(className='highlight',children=[f'{parameters["epsilon_num_lor"]}'])])

# #endregion#

#region# ################### Mu Related Section ##########################
mu_current_index = parameters["mu"]["current_index"]


mu_slider_area_left = html.Div(id='mu-slider-area',className='slider',children=[
        dcc.Slider(
            id='mu-w0-slider',
            min=1,
            max=5,
            step=0.01,
            value=parameters['mu']['w0'][mu_current_index],
            # updatemode='drag',
            marks={
                1:"min:1",
                5:"max:5"
            }
        ),
        html.Div(id='mu-w0-slider-content',children=[],style={"text-align":"center","font-size":"1.5em"}),

        dcc.Slider(
            id='mu-wp-slider',
            min=0,
            max=5,
            step=0.01,
            value=parameters['mu']['wp'][mu_current_index],
            # updatemode='drag',
            marks={
                0:"min:0",
                5:"max:5"
            }
        ),
        html.Div(id='mu-wp-slider-content',children=[],style={"text-align":"center","font-size":"1.5em"}),
        dcc.Slider(
            id='mu-ws-slider',
            min=0,
            max=0.05,
            step=0.001,
            value=parameters['mu']['ws'][mu_current_index],
            # updatemode='drag',
            marks={
                0:"min:0",
                0.05:"max:0.05"
            }
        ),
        html.Div(id='mu-ws-slider-content',children=[],style={"text-align":"center","font-size":"1.5em"})
    ],style={"width":"50%","float":"left"})

mu_slider_area_right=html.Div(className='slider',children=[
    dcc.Slider(
        id='mu-inf-slider',
        min=0,
        max=100,
        step=1,
        value=parameters['mu']['inf'],
        # updatemode='drag',
        marks={
            0:"min:0",
            100:"max:100"
        }
    ),
    html.Div(id='mu-inf-slider-content',style={"text-align":"center","font-size":"1.5em"}),

    dcc.Dropdown(
        id='mu-index-dd',
        options=[{'label': k+1, 'value': k} for k in range(num_lor)],
        value=0
    ),

    html.Button("Add 1 more Lorentzian",id='mu-add-button',className='button',style={"display":"inline-block","font-size":"1.5em","margin":"1em","width":"90%","padding":"0.5em"})
    
],style={"width":"50%","float":"right"})

@app.callback(
    Output(component_id='mu-slider-area',component_property='children'),
    Input(component_id='mu-index-dd',component_property='value')
)
def mu_update_index(selected_index):
    current_index=selected_index
    parameters['mu']['current_index']=current_index
    return html.Div([dcc.Slider(
            id='mu-w0-slider',
            min=1,
            max=5,
            step=0.01,
            value=parameters['mu']['w0'][current_index],
            # updatemode='drag',
            marks={
                1:"min:1",
                5:"max:5"
            }
        ),
        html.Div(id='mu-w0-slider-content',children=[],style={"text-align":"center","font-size":"1.5em"}),

        dcc.Slider(
            id='mu-wp-slider',
            min=0,
            max=5,
            step=0.01,
            value=parameters['mu']['wp'][current_index],
            # updatemode='drag',
            marks={
                0:"min:0",
                5:"max:5"
            }
        ),
        html.Div(id='mu-wp-slider-content',children=[],style={"text-align":"center","font-size":"1.5em"}),
        dcc.Slider(
            id='mu-ws-slider',
            min=0,
            max=0.05,
            step=0.001,
            value=parameters['mu']['ws'][current_index],
            # updatemode='drag',
            marks={
                0:"min:0",
                0.05:"max:0.05"
            }
        ),
        html.Div(id='mu-ws-slider-content',children=[],style={"text-align":"center","font-size":"1.5em"})
    ])

@app.callback(
    [Output(component_id='mu-main-graph',component_property='figure'),
    Output(component_id='mu-w0-slider-content',component_property='children'),
    Output(component_id='mu-wp-slider-content',component_property='children'),
    Output(component_id='mu-ws-slider-content',component_property='children'),
    Output(component_id='mu-inf-slider-content',component_property='children'),
    Output(component_id='mu-index-dd',component_property='options')],
    
    [Input(component_id='mu-w0-slider',component_property='value'),
    Input(component_id='mu-wp-slider',component_property='value'),
    Input(component_id='mu-ws-slider',component_property='value'),
    Input(component_id='mu-inf-slider',component_property='value'),
    Input('mu-add-button', 'n_clicks')]
)
def mu_update_graph(selected_w0,selected_wp,selected_ws,selected_inf,nclick):

    if nclick==None:
        pass

    if nclick != parameters['nclick-mu']:
        parameters['nclick-mu'] = nclick
        parameters['mu'] = parameters['mu_num_lor']+1
        # parameters['epsilon']['current_index'] = parameters['num_lor']+1

        #adding parameters
        w0_new, wp_new,ws_new =  new_param()
        parameters['mu']['w0'].append(w0_new)
        parameters['mu']['wp'].append(wp_new)
        parameters['mu']['ws'].append(ws_new)
    
    current_index = parameters['mu']['current_index']
    parameters['mu']['w0'][current_index]=selected_w0
    parameters['mu']['wp'][current_index]=selected_wp
    parameters['mu']['ws'][current_index]=selected_ws
    parameters['mu']['inf']=selected_inf


    content1 = f'w_0: {selected_w0:.2f}'
    content2 = f'w_p: {selected_wp:.2f}'
    content3 = f'w_s: {selected_ws:.5f}'
    content4 = f'inf: {selected_inf}'


    w0 = torch.tensor(parameters['mu']['w0'])
    wp = torch.tensor(parameters['mu']['wp'])
    ws = torch.tensor(parameters['mu']['ws'])
    eps_inf = torch.tensor(parameters['mu']['inf'])

    num_spectra=2001
    num_lor = parameters['mu_num_lor']

    w0 = w0.unsqueeze(1).expand(num_lor, num_spectra)
    wp = wp.unsqueeze(1).expand_as(w0)
    ws = ws.unsqueeze(1).expand_as(w0)
    w_expand = torch.tensor(w).expand_as(ws)

    num = pow(wp,2)
    denum = pow(w0,2)-pow(w,2)+(1j)*ws*w
    epi_r = eps_inf + torch.sum(torch.div(num,denum),axis=0)

    fig= go.Figure()
    fig.add_trace(go.Scatter(x=list(w),y=list(epi_r.real),name='real part'))
    fig.add_trace(go.Scatter(x=list(w),y=list(epi_r.imag),name='imaginary part'))
    fig.add_trace(go.Scatter(x=[parameters['mu']['w0'][current_index]],y=[0],name='current peak',marker_size=16))
    fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
    ),title=r"$\mu_r$", title_x=0.5,yaxis_range=[-200,200],
    margin=dict(l=20, r=20, t=40, b=20))
    
    return fig, content1, content2, content3, content4, [{'label': k+1, 'value': k} for k in range(num_lor)]


@app.callback(Output(component_id='mu-button-content',component_property='children'), [Input('mu-add-button', 'n_clicks')])
def mu_on_click(nclick):
    return html.Div(children=[f'total number of Lorentzians: ',html.Span(className='highlight',children=[f'{parameters["mu_num_lor"]}'])])

# #endregion#

control_area_children = [
    html.Div(id='epsilon-control',children=[dcc.Graph(id='epsilon-main-graph',figure={}),
    epsilon_slider_area_right,
    epsilon_slider_area_left,
    html.Div(id='epsilon-button-content',style={"display":"block","clear":"both"}),
    html.Div(style={"clear":"both"})]
    ),
    html.Div(id='mu-control',children=[dcc.Graph(id='mu-main-graph',figure={}),
    mu_slider_area_right,
    mu_slider_area_left,
    html.Div(id='mu-button-content',style={"display":"block","clear":"both"}),
    html.Div(style={"clear":"both"})])
]

app.layout=html.Div([
    html.H1("First Dash App - COOL!",style={"text-align":"center"}),
    html.Div(id='control-area',children = control_area_children)
    
])

app.run_server(debug=True)