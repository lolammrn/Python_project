import matplotlib.pyplot as plt
from django.shortcuts import render

import base64
from io import BytesIO
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')


def get_graph_img():
    buf1 = BytesIO()
    plt.savefig(
        buf1, format="png", bbox_inches='tight')
    graph = base64.b64encode(buf1.getvalue()).decode("utf-8")
    buf1.close()
    return graph


def index(request):
    plt.clf()
    x_graph2 = request.GET.get('x_graph2', 'Age')

    df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
    df_viz = df.groupby(['NObeyesdad']).size()

    # graph1
    df_viz.plot(kind='bar')
    graph1 = get_graph_img()

    # graph2
    fig, ax = plt.subplots()
    ax.scatter(df[x_graph2], df["Weight"], c="green",
               alpha=0.5, marker=".", label="Individual")
    ax.set_xlabel(x_graph2)
    ax.set_ylabel("Weight")
    ax.legend()
    graph2 = get_graph_img()

    context = {'graph1': graph1, 'graph2': graph2}
    return render(request, 'main_app/index.html', context)
