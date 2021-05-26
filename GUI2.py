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
    "nclick-mu":None,
    "num_spectra":2001
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
    ])

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
        value=0,style={"color":"black"}
    ),

    html.Button("Add",id='epsilon-add-button',className='button',style={"display":"block","font-size":"1.5em","width":"80%","padding":"0.5em"})
    
])

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
    Output(component_id='epsilon-index-dd',component_property='options'),
    Output(component_id='epsilon-link',component_property='children')],
    
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
    ),yaxis_range=[-200,200],
    margin=dict(l=20, r=20, t=20, b=20))
    
    return fig, content1, content2, content3, content4, [{'label': k+1, 'value': k} for k in range(num_lor)], 'non-sense'


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
    ])

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

    html.Button("Add",id='mu-add-button',className='button',style={"display":"block","font-size":"1.5em","width":"80%","padding":"0.5em"})
    
])

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
    Output(component_id='mu-index-dd',component_property='options'),
    Output(component_id='mu-link',component_property='children')],
    
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
        parameters['mu_num_lor'] = parameters['mu_num_lor']+1
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
    ),yaxis_range=[-200,200],
    margin=dict(l=20, r=20, t=20, b=20))
    
    return fig, content1, content2, content3, content4, [{'label': k+1, 'value': k} for k in range(num_lor)], 'non-sense'


@app.callback(Output(component_id='mu-button-content',component_property='children'), [Input('mu-add-button', 'n_clicks')])
def mu_on_click(nclick):
    return html.Div(children=[f'total number of Lorentzians: ',html.Span(className='highlight',children=[f'{parameters["mu_num_lor"]}'])])

# #endregion#

control_area_children = [
    html.Div(id='epsilon-control',children=[
        dcc.Graph(id='epsilon-main-graph',figure={}),
        html.Div(id='epsilon-control-right',children=[epsilon_slider_area_left,
        epsilon_slider_area_right,
        ]),
        html.Div(id='epsilon-button-content')
    ]),

    html.Div(id='mu-control',children=[
        dcc.Graph(id='mu-main-graph',figure={}),
        html.Div(id='mu-control-right',children=[mu_slider_area_left,
        mu_slider_area_right
        ]),
        html.Div(id='mu-button-content')
    ]),

    #these two divs are just for control 
    html.Div(id='mu-link',style={"display":"none"}),
    html.Div(id='epsilon-link',style={"display":"none"})
]

display_area_children=[
    dcc.Graph(id='n-graph',className='side-plot',figure={}),
    dcc.Graph(id='z-graph',className='side-plot',figure={})
]

main_area_children=[
    dcc.Graph(id='T-graph',figure={}),
    dcc.Graph(id='R-graph',figure={})
]

app.layout=html.Div([
    html.H1("Juncheng App",style={"text-align":"center"}),
    html.Div(id='main-display-area',children=main_area_children),
    html.Div(id='display-area',children=display_area_children),
    html.Div(style={"clear":"both"}),
    html.Div(id='control-area',children = control_area_children)
    
])


@app.callback(
    [Output(component_id='n-graph',component_property='figure'),
    Output(component_id='z-graph',component_property='figure')],
    [Input(component_id='epsilon-main-graph',component_property='figure'),
    Input(component_id='mu-main-graph',component_property='figure'),
    Input(component_id='epsilon-link',component_property='children'),
    Input(component_id='mu-link',component_property='children')]
)
def display_update_graph(fig1,fig2,linktext1,linktext2):
    c=10e8
    e0=(10**7)/(4*np.pi*c**2) 
    m0=4*np.pi*10**(-7)
    
    num_spectra=parameters['num_spectra'] #number of frequency points, which should be 2001
    
    w0e = torch.tensor(parameters['epsilon']['w0'])
    wpe = torch.tensor(parameters['epsilon']['wp'])
    wse = torch.tensor(parameters['epsilon']['ws'])
    eps_infe = torch.tensor(parameters['epsilon']['inf'])

    num_lore = parameters['epsilon_num_lor']
    w0e = w0e.unsqueeze(1).expand(num_lore, num_spectra)
    wpe = wpe.unsqueeze(1).expand_as(w0e)
    wse = wse.unsqueeze(1).expand_as(w0e)
    w_expande = torch.tensor(w).expand_as(wse)

    nume = pow(wpe,2)
    denume = pow(w0e,2)-pow(w,2)+(1j)*wse*w
    epi_r = eps_infe + torch.sum(torch.div(nume,denume),axis=0) #epi_r is episilon relative

    w0m = torch.tensor(parameters['mu']['w0'])
    wpm = torch.tensor(parameters['mu']['wp'])
    wsm = torch.tensor(parameters['mu']['ws'])
    eps_infm = torch.tensor(parameters['mu']['inf'])

    num_lorm = parameters['mu_num_lor']
    #vectorize for broadcasting
    w0m = w0m.unsqueeze(1).expand(num_lorm, num_spectra)
    wpm = wpm.unsqueeze(1).expand_as(w0m)
    wsm = wsm.unsqueeze(1).expand_as(w0m)
    w_expandm = torch.tensor(w).expand_as(wsm)

    numm = pow(wpm,2)
    denumm = pow(w0m,2)-pow(w,2)+(1j)*wsm*w
    mu_r = eps_inf + torch.sum(torch.div(numm,denumm),axis=0)

    epsilon = e0*epi_r
    mu = m0*mu_r

    n = torch.sqrt(epi_r*mu_r).detach()
    z = torch.sqrt(mu/epsilon).detach()

    fig_n= go.Figure()
    fig_n.add_trace(go.Scatter(x=list(w),y=list(n.real),name='real part'))
    fig_n.add_trace(go.Scatter(x=list(w),y=list(n.imag),name='imaginary part'))
    fig_n.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
    ),title=r"n (index of refraction)", title_x=0.5, height=200,
    margin=dict(l=15, r=15, t=30, b=15))

    fig_z= go.Figure()
    fig_z.add_trace(go.Scatter(x=list(w),y=list(z.real),name='real part'))
    fig_z.add_trace(go.Scatter(x=list(w),y=list(z.imag),name='imaginary part'))
    fig_z.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
    ),title=r"z (impandence)", title_x=0.5,height=200,
    margin=dict(l=15, r=15, t=30, b=15))



    return fig_n,fig_z
    




app.run_server(debug=True)