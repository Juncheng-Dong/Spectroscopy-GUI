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

import dash_dangerously_set_inner_html as dashd

#global variable for reset the program
RESET=False



#Store all related parameters in dict 'parameters'
parameters= reset_parameters(None)
#generate equally separated points from 1THz to 5THz
w=np.linspace(freq_low,freq_high,NUM_SPECTRA)

#Initialize the App
app=dash.Dash(__name__)

#region# ################### Epsilon Related Section ##########################
epsilon_slider_area = html.Div(id='epsilon-slider-area',className='slider',
    children = update_epsilon_slider(parameters)
)

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
    [Input(component_id='epsilon-index-dd',component_property='value')]
) #callback for epsilon-index update
def epsilon_slider_update(selected_index):
    global RESET
    if RESET:
        return update_epsilon_slider(parameters)
    
    #if not in RESET, update the current index
    current_index=selected_index
    parameters['epsilon']['current_index']=current_index #update parameters dictionary
    return update_epsilon_slider(parameters)


@app.callback(
    [Output(component_id='epsilon-main-graph',component_property='figure'),
    Output(component_id='epsilon-index-dd',component_property='options'),
    Output(component_id='epsilon-index-dd',component_property='value'),
    Output(component_id='epsilon-link',component_property='children')],
    
    [Input(component_id='epsilon-w0-slider',component_property='value'),
    Input(component_id='epsilon-wp-slider',component_property='value'),
    Input(component_id='epsilon-ws-slider',component_property='value'),
    Input(component_id='epsilon-inf-slider',component_property='value'),
    Input('epsilon-add-button', 'n_clicks'),
    Input('reset-link','children')]
)
def epsilon_update_graph(selected_w0,selected_wp,selected_ws,selected_inf,nclick,resetlink):
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

    if not RESET: #if not in RESET mode, update the parameters dictionary
        parameters['epsilon']['w0'][current_index]=selected_w0
        parameters['epsilon']['wp'][current_index]=selected_wp
        parameters['epsilon']['ws'][current_index]=selected_ws
        parameters['epsilon']['inf']=selected_inf

    if RESET:
        dd_value = 0
    else:
        dd_value = current_index


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

    #plot the center the current Lorentzian
    fig.add_trace(go.Scatter(x=[parameters['epsilon']['w0'][current_index]],y=[0],name='current peak',marker_size=14))
    fig.add_trace(go.Scatter(x=parameters['epsilon']['w0'], y=[0]*num_lor,text=list(range(1,num_lor+1)),marker_size=8,name='Lorentzians',mode='markers',hovertemplate='Lor-index: <br>%{text}'))
    
    fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
    ),yaxis_range=[-200,200],
    margin=dict(l=5, r=5, t=5, b=5))
    
    return fig, [{'label': k+1, 'value': k} for k in range(num_lor)],dd_value, 'non-sense'

@app.callback(Output(component_id='epsilon-button-content',component_property='children'), [Input('epsilon-add-button', 'n_clicks')])
def epsilon_on_click(nclick):
    return html.Div(children=[f'EPSILON #Lorentzians: ',html.Span(className='highlight',children=[f'{parameters["epsilon_num_lor"]}'])])

# #endregion#

#region# ################### Mu Related Section ##########################
mu_slider_area = html.Div(id='mu-slider-area',className='slider',
    children = update_mu_slider(parameters)
)
mu_add_area=html.Div(className='slider',children=[
    dcc.Dropdown(
        id='mu-index-dd',
        options=[{'label': k+1, 'value': k} for k in range(parameters['mu_num_lor'])],
        value=0,style={"color":"black","text-align":"center"},
        clearable=False
    ),
    html.Button("Add",id='mu-add-button',className='button')
])


@app.callback(
    Output(component_id='mu-slider-area',component_property='children'),
    Input(component_id='mu-index-dd',component_property='value')
)
def mu_slider_index(selected_index):
    global RESET
    if RESET:
        return update_mu_slider(parameters)

    current_index=selected_index
    parameters['mu']['current_index']=current_index
    return update_mu_slider(parameters)

@app.callback(
    [Output(component_id='mu-main-graph',component_property='figure'),
    Output(component_id='mu-index-dd',component_property='options'),
    Output(component_id='mu-index-dd',component_property='value'),
    Output(component_id='mu-link',component_property='children'),
    Output(component_id='mu-slider-area',component_property='className')],
    
    [Input(component_id='mu-w0-slider',component_property='value'),
    Input(component_id='mu-wp-slider',component_property='value'),
    Input(component_id='mu-ws-slider',component_property='value'),
    Input(component_id='mu-inf-slider',component_property='value'),
    Input('mu-add-button', 'n_clicks'),
    Input('reset-link','children')]
)
def mu_update_graph(selected_w0,selected_wp,selected_ws,selected_inf,nclick,resetlink):

    if nclick==None:
        pass

    if nclick != parameters['nclick-mu']:
        parameters['nclick-mu'] = nclick
        parameters['mu_num_lor'] = parameters['mu_num_lor']+1

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

    if not RESET: #if not in RESET mode, update the parameters dictionary
        parameters['mu']['w0'][current_index]=selected_w0
        parameters['mu']['wp'][current_index]=selected_wp
        parameters['mu']['ws'][current_index]=selected_ws
        parameters['mu']['inf']=selected_inf

    if RESET:
        dd_value = 0
    else:
        dd_value = current_index



    w0 = torch.tensor(parameters['mu']['w0'])
    wp = torch.tensor(parameters['mu']['wp'])
    ws = torch.tensor(parameters['mu']['ws'])
    eps_inf = torch.tensor(parameters['mu']['inf'])

    num_spectra=NUM_SPECTRA
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
        fig.add_trace(go.Scatter(x=parameters['mu']['w0'], y=[0]*num_lor,text=list(range(1,num_lor+1)),marker_size=8,name='Lorentzians',mode='markers',hovertemplate='Lor-index: <br>%{text}'))
    fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
    ),  
    margin=dict(l=20, r=10, t=10, b=20))

    if num_lor == 0:
        className = 'slider invisible'
    else:
        className = 'slider'
    
    return fig, [{'label': k+1, 'value': k} for k in range(num_lor)], dd_value, 'non-sense',className


@app.callback(Output(component_id='mu-button-content',component_property='children'), [Input('mu-add-button', 'n_clicks')])
def mu_on_click(nclick):
    return html.Div(children=[f'MU #Lorentzians: ',html.Span(className='highlight',children=[f'{parameters["mu_num_lor"]}'])])
    # return dashd.DangerouslySetInnerHTML(
    #     f'''<div>&mu; #Lorentzians: <span class='highlight'>{parameters["mu_num_lor"]} <span> </div>'''
    # )

# #endregion#

#region# ################### Main Plot/Load/Reset #########################
@app.callback(
    [Output('thickness-slider-content','children'),
    Output('thickness-link','children')],
    [Input(component_id='thickness-slider',component_property='value'),
    Input('reset-link','children')]
)
def thickness_update(slider_value,resetlink):
    if RESET:
        parameters['thickness']=thickness
        return_value = thickness
    else:
        parameters['thickness'] = slider_value
        return_value = slider_value
    
    return f'Material Thickness: {return_value} nm','nonsense'

@app.callback(
    [Output(component_id='n-graph',component_property='figure'),
    Output(component_id='z-graph',component_property='figure'),
    Output(component_id='T-graph',component_property='figure')],
    
    [Input(component_id='epsilon-link',component_property='children'),
    Input(component_id='mu-link',component_property='children'),
    Input(component_id='upload-link',component_property='children'),
    Input(component_id='thickness-link',component_property='children')]
)
def display_update_graph(linktext1,linktext2,linktext3,linktext4):
    c=3e-4
    e0=9.85e-12
    m0=4*np.pi*10**(-7)
    d = parameters['thickness']*1e-6
    
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

    k=100*torch.tensor(w)*2*np.pi*n/c #wn/c
    t = 1/(torch.cos(k*d)-1j/2*(z+torch.pow(z,-1))*torch.sin(k*d))
    r = -1j/2*(z-torch.pow(z,-1)*torch.sin(k*d))*t

    T = t.real**2 + t.imag**2
    T2= torch.exp(-2*k.imag*d)* ( (n/mu_r).real * torch.abs((2*mu_r)/(n+mu_r))**2)**2

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
    fig_T.add_trace(go.Scatter(x=list(w),y=list(T2.detach()),name='T2'))
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

    global RESET
    RESET=False

    return fig_n,fig_z,fig_T
    
@app.callback(
    [Output('output-data-upload', 'children'),
    Output('upload-link','children')],
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_load(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        data,children = parse_contents(list_of_contents, list_of_names, list_of_dates)
        # [parse_contents(c, n, d) for c, n, d in
        #     zip(list_of_contents, list_of_names, list_of_dates)]
        parameters['target_spectrum'] = data[0]
        return children,'nonsense'
    else:
        return None,'nonsense'

@app.callback(
    Output('reset-link','children'),
    Input('reset-area','n_clicks')
)
def update_reset(reset_nclicks):
    global RESET
    print("reset_nclicks:",reset_nclicks)
    if reset_nclicks != parameters['nclick-reset']:
        RESET=True
        temp_parameters = reset_parameters(reset_nclicks,parameters)

        for key in temp_parameters:
            parameters[key] = temp_parameters[key]
    
    return 'nonsense'
# #endregion#

#region#  ####################### App Layout Section ########################
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
    dcc.Graph(id='T-graph',figure={}),
    dcc.Slider(
        id='thickness-slider',
        min=thickness_low,
        max=thickness_high,
        step=thickness_step,
        value=parameters['thickness'],
        # updatemode='drag',
        marks={
            thickness_low:{"label":f"{thickness_low} ",'style': {'color': 'white'}},
            thickness_high:{"label":f"{thickness_high} ",'style': {'color': 'white'}}
        }
    ),
    html.Div(id='thickness-slider-content',children=[]),
    html.Div(id='thickness-link',style={"display":"none"}),
]

title_bar_children=[
    upload_component,
    
    html.H1(children=[html.Span("Juncheng",style={"text-decoration":"underline"}),"App"],style={"text-align":"center"}),
    reset_component,
    # dashd.DangerouslySetInnerHTML('''<div>&#9211; </div>'''),
    html.Div(id='upload-link',style={"display":"none"}),
    html.Div(id='reset-link',style={"display":"none"})
]

app.layout=html.Div([
    html.Div(id='title-bar',children=title_bar_children),
    html.Div(id='main-display-area',children=main_area_children),
    html.Div(id='display-area',children=display_area_children),html.Div(style={"clear":"both"}),
    html.Div(id='control-area',children = control_area_children)
])
# #endregion#

#Finally! Run the Server
app.run_server(debug=True)