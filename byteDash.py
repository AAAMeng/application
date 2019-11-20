#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   byteDistribDash.py
@Contact :   zhumeng@bupt.edu.cn
@License :   (C)Copyright 2019 zhumeng

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/19 15:24      xm         1.0          None
"""
import random
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from preAnalysis import read_from_csv

app = dash.Dash(__name__)

mydata = read_from_csv()
x_data = list(mydata.keys())

app.layout = html.Div([
    html.Div([html.H1("Byte Distribution for Different Application")], style={"textAlign": "center"}),
    dcc.Graph(id="my-graph"),
    html.Div([dcc.Slider(id="my-slider", min=1, max=18, marks={i: '{}'.format(10 * i) for i in range(1, 19)},
                         step=0.1, value=1, included=False, updatemode='drag')],
             style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "60%"}, ),
    html.Div(id='current_byte', style={'margin-top': 20})
])


@app.callback(
    [
        Output('my-graph', 'figure'),
        Output('current_byte', 'children')
    ],
    [Input('my-slider', 'value')])
def update_figure(input_value):
    traces = []
    for xd in x_data:
        traces.append(
            go.Box(
                marker=dict(
                    outliercolor='rgba(219, 64, 82, 0.6)',
                    line=dict(
                        outliercolor='rgba(219, 64, 82, 0.6)',
                        outlierwidth=2)),
                line_width=1))
    return {"data": traces,
            "layout": go.Layout(title='Byte Distribution(v.1.3)',
                                # yaxis=dict(
                                #     autorange=True,
                                #     showgrid=True,
                                #     zeroline=True,
                                #     dtick=5,
                                #     gridcolor='rgb(255, 255, 255)',
                                #     gridwidth=5,
                                #     zerolinecolor='rgb(255, 255, 255)',
                                #     zerolinewidth=1,
                                # ),
                                # margin=dict(
                                #     l=30,
                                #     r=30,
                                #     b=80
                                # ),
                                showlegend=True,
                                autosize=True, )
            }, '(Start from 1)\nCurrent Byte: ' + str(int(input_value * 10))


if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
