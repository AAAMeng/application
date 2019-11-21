#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   lengthDash.py    
@Contact :   zhumeng@bupt.edu.cn
@License :   (C)Copyright 2019 zhumeng

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/20 14:44      xm         1.0          None
"""
import random
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.figure_factory as ff
from dash.dependencies import Input, Output
from preAnalysis import read_from_txt

app = dash.Dash(__name__)

mydata = read_from_txt(fillna=False)
x_data = list(mydata.keys())
y_data = list(mydata.get(app).count(axis=1) for app in x_data)
app.layout = html.Div([
    html.Div([html.H1("Packet Length for Different Application")], style={"textAlign": "center"}),
    # dcc.Graph(id="my-graph",figure = fig),
    dcc.Graph(
        id='my-graph',
        figure=ff.create_distplot(y_data, x_data, bin_size=10)
    ),
    dcc.Graph(
        id='my-graph-2',
        figure=ff.create_distplot(y_data, x_data, bin_size=5)
    )
])


if __name__ == '__main__':
    app.run_server(debug=True, port=8081)
