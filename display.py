import os
import json
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

def display_favorability_scores(data_path: str):
    # Load data from JSON file
    if not data_path:
        data_path = os.path.join(os.path.dirname(__file__), "data", "test_aggregations.json")
    with open(data_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    names = [item.get("name") for item in items]
    scores = [float(item.get("score", 0)) for item in items]

    def get_color_label(score: float):
        if score <= 0.25:
            return 'Hated', 'red'
        elif score <= 0.40:
            return 'Disliked', 'orange'
        elif score <= 0.80:
            return 'Admired', 'lightgreen'
        else:
            return 'Loved', 'darkgreen'

    fig = go.Figure()
    for i, item in enumerate(items):
        value = float(item.get("score", 0))
        label, color = get_color_label(value)
        fig.add_trace(
            go.Indicator(
                mode = "gauge+number",
                value = value,
                title = {'text': item.get("name")},
                gauge= {
                    'axis': {'range': [0, 1]},
                    'bar': {'color': color}
                },
                domain = {'row': i//4, 'column': i%4}
            ),
        )

    fig.update_layout(
        grid = {'rows': int(len(names)//4)+1, 'columns': 4, 'pattern': "independent"},
        title = {
            'text': "Favorability Scores of Athletes",
            'x':0.2,
            'xanchor': 'center'
        }
        )
    return fig

