from setting import *
from helper import *

import torch

import numpy as np
import pandas as pd

import plotly.offline as pyo
import plotly.graph_objects as go
import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

def init_param(num_lor):
    w0 = torch.tensor(np.random.uniform(freq_low, freq_high, num_lor), requires_grad=True)
    wp = torch.tensor(np.random.uniform(wp_freq_low, wp_freq_high, num_lor), requires_grad=True)
    ws = torch.tensor(np.random.uniform(ws_freq_low, ws_freq_high, num_lor), requires_grad=True)
    eps_inf = torch.tensor(5., requires_grad=True)
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
w0,wp,ws,eps_inf,d=init_param(num_lor_init_epsilon) 
w0[0]=3
wp[0]=3
ws[0]=0.01


w0m,wpm,wsm,eps_infm,dm = init_param(num_lor_init_mu)

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
    "epsilon_num_lor":num_lor_init_epsilon,
    "mu_num_lor":0,
    "nclick-epsilon":None,
    "nclick-mu":None,
    "num_spectra":NUM_SPECTRA,
    "target_spectrum":None
}
#generate equally separated points from 1THz to 5THz
w=np.linspace(freq_low,freq_high,NUM_SPECTRA)

#Initialize the App
app=dash.Dash(__name__)

#region# ################### Epsilon Related Section ##########################
epsilon_slider_area = update_epsilon_slider(parameters)
epsilon_add_area=html.Div(className='slider',children=[

    dcc.Dropdown(
        id='epsilon-index-dd',
        options=[{'label': k+1, 'value': k} for k in range(parameters['epsilon_num_lor'])],
        value=0,style={"color":"black","text-align":"center"},
        clearable=False
    ),

    html.Button("Add",id='epsilon-add-button',className='button')
    
])

@app.callback(
    Output(component_id='epsilon-slider-area',component_property='children'),
    Input(component_id='epsilon-index-dd',component_property='value')
)
def epsilon_slider_update(selected_index):
    current_index=selected_index
    parameters['epsilon']['current_index']=current_index #update parameters dictionary
    return update_epsilon_slider(parameters)


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
        #update parameters dictionary
        parameters['nclick-epsilon'] = nclick
        parameters['epsilon_num_lor'] = parameters['epsilon_num_lor']+1

        #adding a new set of Lorentzian parameters
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
    denum = pow(w0,2)-pow(w_expand,2)-(1j)*ws*w_expand
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
mu_slider_area = update_mu_slider(parameters)
mu_add_area=html.Div(className='slider',children=[

    dcc.Dropdown(
        id='mu-index-dd',
        options=[{'label': k+1, 'value': k} for k in range(parameters['mu_num_lor'])],
        value=0,
        clearable=False
    ),

    html.Button("Add",id='mu-add-button',className='button')
    
])

@app.callback(
    Output(component_id='mu-slider-area',component_property='children'),
    Input(component_id='mu-index-dd',component_property='value')
)
def mu_update_index(selected_index):
    current_index=selected_index
    parameters['mu']['current_index']=current_index
    return update_mu_slider(parameters)

@app.callback(
    [Output(component_id='mu-main-graph',component_property='figure'),
    Output(component_id='mu-w0-slider-content',component_property='children'),
    Output(component_id='mu-wp-slider-content',component_property='children'),
    Output(component_id='mu-ws-slider-content',component_property='children'),
    Output(component_id='mu-inf-slider-content',component_property='children'),
    Output(component_id='mu-index-dd',component_property='options'),
    Output(component_id='mu-link',component_property='children'),
    Output(component_id='mu-slider-area',component_property='className')],
    
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
        if parameters['mu_num_lor']==1:
            pass
        else:
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
    denum = pow(w0,2)-pow(w,2)-(1j)*ws*w
    epi_r = eps_inf + torch.sum(torch.div(num,denum),axis=0)

    if num_lor == 0:
        epi_r = np.array([1.0]*num_spectra)

    fig= go.Figure()
    fig.add_trace(go.Scatter(x=list(w),y=list(epi_r.real),name='real part'))
    fig.add_trace(go.Scatter(x=list(w),y=list(epi_r.imag),name='imaginary part'))
    #adding peaks
    if num_lor == 0:
        pass
    else:
        fig.add_trace(go.Scatter(x=[parameters['mu']['w0'][current_index]],y=[0],name='current peak',marker_size=16))
    fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
    ),  ##yaxis_range=[-200,200],
    margin=dict(l=20, r=20, t=20, b=20))

    'a'
    if num_lor == 0:
        className = 'slider invisible'
    else:
        className = 'slider'
    
    return fig, content1, content2, content3, content4, [{'label': k+1, 'value': k} for k in range(num_lor)], 'non-sense',className


@app.callback(Output(component_id='mu-button-content',component_property='children'), [Input('mu-add-button', 'n_clicks')])
def mu_on_click(nclick):
    return html.Div(children=[f'total number of Lorentzians: ',html.Span(className='highlight',children=[f'{parameters["mu_num_lor"]}'])])

# #endregion#

control_area_children = [
    html.Div(id='epsilon-control',children=[
        dcc.Graph(id='epsilon-main-graph',figure={}),
        html.Div(id='epsilon-control-right',children=[epsilon_slider_area,
        epsilon_add_area
        ]),
        html.Div(id='epsilon-button-content')
    ]),

    html.Div(id='mu-control',children=[
        dcc.Graph(id='mu-main-graph',figure={}),
        html.Div(id='mu-control-right',children=[mu_slider_area,
        mu_add_area
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
    dcc.Graph(id='T-graph',figure={})
]

title_bar_children=[
    upload_component,
    
    html.H1(children=[html.Span("Juncheng",style={"text-decoration":"underline"}),"App"],style={"text-align":"center"}),
    reset_component,
    html.Div(id='upload-link',style={"display":"none"})
]

app.layout=html.Div([
    html.Div(id='title-bar',children=title_bar_children),
    html.Div(id='main-display-area',children=main_area_children),
    html.Div(id='display-area',children=display_area_children),html.Div(style={"clear":"both"}),
    html.Div(id='control-area',children = control_area_children)
])


@app.callback(
    [Output(component_id='n-graph',component_property='figure'),
    Output(component_id='z-graph',component_property='figure'),
    Output(component_id='T-graph',component_property='figure')],
    
    [Input(component_id='epsilon-link',component_property='children'),
    Input(component_id='mu-link',component_property='children'),
    Input(component_id='upload-link',component_property='children')]
)
def display_update_graph(linktext1,linktext2,linktext3):
    c=3e-4
    e0=9.85e-12
    m0=4*np.pi*10**(-7)
    d = 10e-6
    
    num_spectra=parameters['num_spectra'] #number of frequency points, which should be 2001
    
    w0e = torch.tensor(parameters['epsilon']['w0'])
    wpe = torch.tensor(parameters['epsilon']['wp'])
    wse = torch.tensor(parameters['epsilon']['ws'])
    eps_infe = torch.tensor(parameters['epsilon']['inf'])

    num_lore = parameters['epsilon_num_lor']
    w0e = 2*np.pi*w0e.unsqueeze(1).expand(num_lore, num_spectra)
    wpe = 2*np.pi*wpe.unsqueeze(1).expand_as(w0e)
    wse = 2*np.pi*wse.unsqueeze(1).expand_as(w0e)
    w_expande = 2*np.pi*torch.tensor(w).expand_as(wse)

    nume = pow(wpe,2)
    denume = pow(w0e,2)-pow(w_expande,2)-(1j)*wse*w_expande
    epi_r = eps_infe + torch.sum(torch.div(nume,denume),axis=0) #epi_r is episilon relative

    w0m = torch.tensor(parameters['mu']['w0'])
    wpm = torch.tensor(parameters['mu']['wp'])
    wsm = torch.tensor(parameters['mu']['ws'])
    eps_infm = torch.tensor(parameters['mu']['inf'])

    num_lorm = parameters['mu_num_lor']
    #vectorize for broadcasting
    w0m = 2*np.pi*w0m.unsqueeze(1).expand(num_lorm, num_spectra)
    wpm = 2*np.pi*wpm.unsqueeze(1).expand_as(w0m)
    wsm = 2*np.pi*wsm.unsqueeze(1).expand_as(w0m)
    w_expandm = 2*np.pi*torch.tensor(w).expand_as(wsm)

    numm = pow(wpm,2)
    denumm = pow(w0m,2)-pow(w,2)-(1j)*wsm*w_expandm
    mu_r = eps_infm + torch.sum(torch.div(numm,denumm),axis=0)

    if parameters['mu_num_lor']==0:
        mu_r = np.array([1.0]*num_spectra)

    epsilon = e0*epi_r
    mu = m0*mu_r

    n = torch.sqrt(epi_r*mu_r).detach()
    z = torch.sqrt(mu/epsilon).detach()

    k=torch.tensor(w)*2*np.pi*n/c #wn/c
    t = 1/(torch.cos(k*d)-1j/2*(z+torch.pow(z,-1))*torch.sin(k*d))
    r = -1j/2*(z-torch.pow(z,-1)*torch.sin(k*d))*t

    T = t.real**2 + t.imag**2

    fig_n= go.Figure()
    fig_n.add_trace(go.Scatter(x=list(w),y=list(n.real),name='real part'))
    fig_n.add_trace(go.Scatter(x=list(w),y=list(n.imag),name='imaginary part'))
    fig_n.update_layout(
        showlegend=False,
        title=r"n-index of refraction", title_x=0.5, height=200,
        margin=dict(l=15, r=15, t=30, b=15)
    )

    fig_z= go.Figure()
    fig_z.add_trace(go.Scatter(x=list(w),y=list(z.real),name='real part'))
    fig_z.add_trace(go.Scatter(x=list(w),y=list(z.imag),name='imaginary part'))
    fig_z.update_layout(
        showlegend=False,
        title=r"z-impandence", title_x=0.5,height=200,
        margin=dict(l=15, r=15, t=30, b=15)
    )

    fig_T= go.Figure()
    fig_T.add_trace(go.Scatter(x=list(w),y=list(T.detach()),name='constructed'))
    if not parameters['target_spectrum'] is None:
        fig_T.add_trace(go.Scatter(x=list(w),y=list(parameters['target_spectrum']),name='target',
        line = dict(color='grey', width=2, dash='dash')))
    fig_T.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
    ),title="Spectrum - T", title_x=0.5,height=400,yaxis_range=[-0.05,1],
    margin=dict(l=15, r=15, t=30, b=15))

    return fig_n,fig_z,fig_T
    
@app.callback(
    [Output('output-data-upload', 'children'),
    Output('upload-link','children')],
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        data,children = parse_contents(list_of_contents, list_of_names, list_of_dates)
        # [parse_contents(c, n, d) for c, n, d in
        #     zip(list_of_contents, list_of_names, list_of_dates)]
        parameters['target_spectrum'] = data[0]
        print(data[0])
        return children,'nonsense'
    else:
        return None,'nonsense'

#Finally! Run the Server
app.run_server(debug=True)