import json
import numpy as np
from flask import Flask, request, jsonify
import threading
import time
import webbrowser
import plotly.graph_objects as go

from sentiment import predict_sentiment
from scraper import get_reddit_comments


''' This is the data we got with running our algorithm.
It takes a while for the program to run so we wanted a few athletes to pre-populate
the webpage.

It is possible that the api will look at different comment counts and a 
different score if the user types one of these names to add it.
'''
preloaded = [
    {"name": "LeBron James", "score": 0.6195652173913043, "count": 92},
    {"name": "Lionel Messi", "score": 0.7176470588235294, "count": 85},
    {"name": "Stephen Curry", "score": 0.5714285714285714, "count": 84}
]

athletes = [p["name"] for p in preloaded]
scores = {p["name"]: p["score"] for p in preloaded}
counts = {p["name"]: p["count"] for p in preloaded}


loading = None

# This function gets called when a new athletes name is added
def compute_likeability(name):
    global loading
    loading = name

    comments = get_reddit_comments(name)

    if len(comments) == 0:
        loading = None
        return 0.5, 0

    preds = [predict_sentiment(c) for c in comments]
    score = float(np.mean(preds))

    loading = None
    return score, len(comments)


# Function thats creates the gauges for the data
def build_plot_json():
    fig = go.Figure()

    def col(s):
        if s <= 0.25: return "red"
        if s <= 0.40: return "orange"
        if s <= 0.80: return "lightgreen"
        return "darkgreen"

    names = list(scores.keys())

    for i, name in enumerate(names):
        # Loops through the list of athlete names to create gauges using go
        s = scores[name]
        c = counts.get(name, 0)

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=s,
                title={"text": f"{name} ({c} comments)"},
                gauge={"axis": {"range": [0, 1]}, "bar": {"color": col(s)}},
                domain={"row": i // 3, "column": i % 3}
            )
        )

    # Creats the setup for each gauge
    fig.update_layout(
        grid={"rows": (len(names) // 3) + 1, "columns": 3, "pattern": "independent"},
        height= 450 + 150 * (len(names) // 3),
        title={"text": "Athlete Favorability Dashboard"}
    )

    return fig.to_dict()



# Creates the local flask app that
app = Flask(__name__)

# HTML of the app
@app.route("/")
def index():
    return """
    <html>
    <body style='font-family:Arial; margin:40px;'>

    <h1>Athlete Favorability Dashboard</h1>

    <div id='loading'
         style='font-size:20px;color:gray;display:none;margin-bottom:10px;'>
         Loading athlete (This can take up to 30 seconds)
    </div>

    <input id='name' type='text' placeholder='Add athlete'
           style='width:260px;height:30px;font-size:16px;'/>
    <button onclick='addAth()' style='height:34px;'>Add</button>

    <div id='plot'></div>

    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

    <script>

        function addAth() {
            let n = document.getElementById("name").value;
            if (n.length < 2) return;

            document.getElementById("loading").innerText =
                "Loading " + n + " ";
            document.getElementById("loading").style.display = "block";

            fetch("/add?name=" + encodeURIComponent(n));
        }

        async function update() {

            let loadState = await fetch("/loading");
            let l = await loadState.text();

            if (l === "none") {
                document.getElementById("loading").style.display = "none";
            } else {
                document.getElementById("loading").innerText =
                    "Loading " + l + " ";
                document.getElementById("loading").style.display = "block";
            }

            let res = await fetch("/data");
            let chart = await res.json();
            Plotly.react("plot", chart.data, chart.layout);
        }

        setInterval(update, 1500);
        update();

    </script>
    </body>
    </html>
    """

# Function that is used to add athlete names when typed in
@app.route("/add")
def add():
    global athletes

    name = request.args.get("name")

    if name and name not in athletes:
        athletes.append(name)
        threading.Thread(target=add_single, args=(name,)).start()

    return "OK"


# Called when adding an athlete
def add_single(name):
    global scores, counts
    s, c = compute_likeability(name)
    scores[name] = s
    counts[name] = c


@app.route("/loading")
def loading_state():
    return loading if loading else "none"


@app.route("/data")
def data():
    return jsonify(build_plot_json())


# Launches server
if __name__ == "__main__":
    threading.Timer(1, lambda:
        webbrowser.open("http://127.0.0.1:5000")
    ).start()

    app.run(port=5000, debug=False, threaded=True)
