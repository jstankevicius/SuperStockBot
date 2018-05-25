# A script to launch a stock visualization app based on Dash. While this only provides
# a candlestick chart of a certain stock at a certain time, we hope to integrate
# some model monitoring functionality to make this a more robust and interactive version
# of Tensorboard.

import dash
import csv
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from datetime import datetime
from data_loader import manager
import traceback
import numpy as np
import time
from training import training_session
from threading import Thread

# Construction of the app:
app = dash.Dash()

app.layout = html.Div([
    html.H1("StockBot Data Visualizer",
            style={"font-size": "36",
                   "margin-left": "10px",
                   "font-weight": "bolder",
                   "font-family": "Arial",
                   "color": "rgba(117, 117, 117, 0.95)",
                   "margin-bottom": "20px"}),

    html.Div([
        html.Label(
            "Symbol (TSLA, MSFT, etc)",
            style={"font-size": "14",
                   "margin-left": "10px",
                   "font-family": "Arial",
                   "display": "inline-block"}
        ),

        html.Label(
            "Date (YYYY-MM-DD)",
            style={"font-size": "14",
                   "margin-left": "60px",
                   "font-family": "Arial",
                   "display": "inline-block"}
        )
    ]),

    html.Div([
        dcc.Input(
            id="symbol-input",
            type="text",
            style={"width": "200px",
                   "margin-left": "10px",
                   "display": "inline-block"}
        ),
        dcc.Input(
            id="date-input",
            type="text",
            style={"width": "200px",
                   "margin-left": "10px",
                   "display": "inline-block"}
        )
    ]),
    
    html.Div(id="graph"),

    dcc.Interval(
        id="interval",
        interval=1*1000, #ms
        n_intervals=0
    ),

    html.Div(id="profit-graph"),
    
    # First row of analytics
    html.Div(id="first-row"),

    # Second row of analytics
    html.Div(id="second-row")
])

# This assumes that strings are provided in YYYY-MM-DD format.
def ToD(s):
    # print(s)
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

# basically a copy of jim's method in finance minus csv file reading
# (it's faster this way for single days)
# also for whatever reason everything gets read as a string first
def sma(closing, n):
    values = []
    c = [float(p) for p in closing]
    for i in range(n, len(c)):
        values.append(sum(c[(i-n):i])/n)

    return values

@app.callback(
    Output("graph", "children"),
    [Input("symbol-input", "value"),
     Input("date-input", "value")])
def update_graph(symbol, date):
    t = []
    o = []
    h = []
    l = []
    c = []
    filepath = "data//intraday//" + symbol + ".csv"

    elements = []
    try:
        with open(filepath, "r") as file:

            reader = csv.reader(file)
            for row in reader:
                if date == row[0]:

                    # extremely dumb fix
                    t.append(ToD(row[0] + " " + row[1]))

                    # then append the rest of the values
                    o.append(row[2])
                    h.append(row[3])
                    l.append(row[4])
                    c.append(row[5])

            candlestick = {
                    "x": t,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "type": "candlestick",
                    "name": symbol,
                    "legendgroup": symbol,
                    "increasing": {"line": {"color": "rgba(255, 0, 0, 0.95)"}},
                    "decreasing": {"line": {"color": "rgba(0, 0, 255, 0.95)"}}
            }

            # i need to fix this
            sma_graph = {
                "x": t[10:],
                "y": sma(c, 10),
                "type": "scatter",
                "mode": "lines",
                "line": {"width": 1, "color": "rgba(0, 0, 0, 1)"},
                "legendgroup": symbol,
                "name": "SMA"
            }

            graph = dcc.Graph(
                id=symbol,
                figure={
                    "data": [candlestick, sma_graph],
                    "layout": {
                        "margin": {"b": 0, "r": 10, "l":60, "t":0},
                        "legend": {"x": 0}
                    }
                }
            )

            return graph
    except OSError:
        return "INVALID PATH: " + filepath
    except:
        return str(traceback.format_exc())


###################################################################################
# TRAINING RUN
agent_stats = {
    
    "classical_brownian": {
        "loss": [],
        "min_return": [],
        "max_return": [],
        "avg_return": [],
        "buy_success": [],
        "sell_success": [],
        "win_rate": [],
        "activity": [],
        "profit": []
    },
    
    "geometric_brownian": {
        "loss": [],
        "min_return": [],
        "max_return": [],
        "avg_return": [],
        "buy_success": [],
        "sell_success": [],
        "win_rate": [],
        "activity": [],
        "profit": []
    },

    "merton_jump_diffusion": {
        "loss": [],
        "min_return": [],
        "max_return": [],
        "avg_return": [],
        "buy_success": [],
        "sell_success": [],
        "win_rate": [],
        "activity": [],
        "profit": []
    },

    "heston_stochastic": {
        "loss": [],
        "min_return": [],
        "max_return": [],
        "avg_return": [],
        "buy_success": [],
        "sell_success": [],
        "win_rate": [],
        "activity": [],
        "profit": []
    }
}

line_colors = {
    "classical_brownian": "rgba(0, 153, 136, 1)",
    "geometric_brownian": "rgba(0, 119, 187, 1)",
    "merton_jump_diffusion": "rgba(255, 112, 67, 1)",
    "heston_stochastic": "rgba(238, 51, 119, 1)"
}

@app.callback(
    Output("profit-graph", "children"),
    [Input("interval", "n_intervals")])
def update_profit_graph(n):
    try:
        data = []
        for key in agent_stats.keys():
            agent_profit = agent_stats[key]["profit"]
            iterations = []
            for i in range(len(agent_profit)):
                iterations.append(i + 1)

            reward_graph = {
                "x": iterations,
                "y": agent_profit,
                "type": "scatter",
                "mode": "lines",
                "line": {"width": 2, "color": line_colors[key]},
                "legendgroup": key,
                "name": key
            }

            data.append(reward_graph)

        graph = dcc.Graph(
            id="profit",
            figure = {
                "data": data,
                "layout": {
                    "margin": {"t": 20, "b": 20, "r": 40, "l": 40}
                }
            }
        )

        return graph
    except:
        return str(traceback.format_exc())

@app.callback(
    Output("first-row", "children"),
    [Input("interval", "n_intervals")])
def update_first_row(n):
    try:
        # This contains the actual DCC objects.
        divs = []

        # Quite ugly. Whatever.
        first_row_stats = ("loss", "min_return", "max_return", "avg_return")

        for i in range(len(first_row_stats)):

            # We DEFINITELY want to do this in order.
            stat = first_row_stats[i]

            # This contains the graphs for the metric we are reading.
            agent_specific_data = []

            # We iterate through every agent and collect the data specific to the agent.
            for key in agent_stats.keys():

                # Stats specific to an agent.
                agent_specific_stats = agent_stats[key]

                data = agent_specific_stats[stat]
                iterations = []
                for i in range(len(data)):
                    iterations.append(i + 1)

                agent_specific_graph = {
                    "x": iterations,
                    "y": data,
                    "type": "scatter",
                    "mode": "lines",
                    "line": {"width": 2, "color": line_colors[key]}
                }

                agent_specific_data.append(agent_specific_graph)

            graph = dcc.Graph(
                id=stat,
                figure = {
                    "data": agent_specific_data,
                    "layout": {
                        "margin": {"t": 20, "b": 20, "r": 40, "l": 40},
                        "showlegend": False
                    }
                }
            )

            label = html.Label(stat, style={"margin-top": "20px", "font-family": "Arial"})
            
            div = html.Div(children=[label, graph], style={"display": "inline-block", "width": "25%", "margin-top": "25px", "margin-bottom": "25px"})
            divs.append(div)
            
        return divs
    except:
        return [str(traceback.format_exc())]
     
@app.callback(
    Output("second-row", "children"),
    [Input("interval", "n_intervals")])
def update_second_row(n):
    try:
        # This contains the actual DCC objects.
        divs = []

        # Quite ugly. Whatever.
        second_row_stats = ("buy_success", "sell_success", "win_rate", "activity")

        for i in range(len(second_row_stats)):

            # We DEFINITELY want to do this in order.
            stat = second_row_stats[i]

            # This contains the graphs for the metric we are reading.
            agent_specific_data = []

            # We iterate through every agent and collect the data specific to the agent.
            for key in agent_stats.keys():

                # Stats specific to an agent.
                agent_specific_stats = agent_stats[key]

                data = agent_specific_stats[stat]
                iterations = []
                for i in range(len(data)):
                    iterations.append(i + 1)
                    
                agent_specific_graph = {
                    "x": iterations,
                    "y": data,
                    "type": "scatter",
                    "mode": "lines",
                    "line": {"width": 2, "color": line_colors[key]}
                }

                agent_specific_data.append(agent_specific_graph)

            graph = dcc.Graph(
                id=stat,
                figure = {
                    "data": agent_specific_data,
                    "layout": {
                        "margin": {"t": 0, "b": 20, "r": 40, "l": 40},
                        "showlegend": False
                    }
                }
            )
            
            label = html.Label(stat, style={"margin-top": "50px", "font-family": "Arial"})
            
            div = html.Div(children=[label, graph], style={"display": "inline-block", "width": "25%", "margin-top": "25px", "margin-bottom": "50px"})
            divs.append(div)

        return divs
    except:
        return str(traceback.format_exc())

# Instantiate a single training session
tr = training_session()
tr.run()

# alright boys this is gonna get gross
window = 100
iteration_count = 1

# 50,000 iterations
def train():
    for i in range(100):
        for k in range(50):
            tr.run()
            
            # Update stats for all agents:
            stats = tr.get_stats()
            for agent in stats.keys():
                agent_specific_stats = stats[agent]
                for key in agent_specific_stats.keys():
                    data_point = agent_specific_stats[key]
                    agent_stats[agent][key].append(data_point)

                    if len(agent_stats[agent][key]) > window:
                        del agent_stats[agent][key][0]

            global iteration_count  
            iteration_count += 1
            tr.reset()

        # Just test one stock.
        rewards = tr.test(symbol_list=["AAPL"])
        tr.save_agents(rewards, prefix=str(iteration_count) + "_")
        for agent in rewards.keys():
            reward = rewards[agent]
            agent_stats[agent]["profit"].append(reward)

    final_rewards = tr.test()
    tr.save_agents(final_rewards, prefix="final_")
            

###################################################################################    
# Start the actual program
t = Thread(target=train)
t.start()
app.run_server(debug=True)
