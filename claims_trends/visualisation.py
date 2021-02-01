# %% Should be able to set these functions up such that they work within a class.
# Need to have a think about how we export these or create a dashboard for eagle

# Maybe use flask endpoint?


# %% Package Imports

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io


# %% Plotting functions

# ~ arg: doc_term_matrix:pd.DataFrame

def plot_linegraph():

    # plot chart, use dummy for now
    sns.set()

    flights = sns.load_dataset('flights')
    sns.lineplot(data=flights, x='year', y='passengers', hue='month', style='month')


    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return bytes_image


def plot_pairplot():

    # plot chart, use dummy for now
    sns.set()

    df = sns.load_dataset('iris')
    sns.pairplot(df, hue='species', size=2.5)

    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return bytes_image


# %% Flask - this application works quite nicely for an easy way to expose the png

from flask import Flask, send_file, make_response

app = Flask(__name__)

@app.route('/test/plot1', methods=['GET'])
def lineplot1():
    bytes_object = plot_linegraph()

    return send_file(bytes_object,
                     attachment_filename='plot.png',
                     mimetype='image/png')

@app.route('/test/plot2', methods=['GET'])
def lineplot2():
    bytes_object = plot_pairplot()

    return send_file(bytes_object,
                     attachment_filename='plot.png',
                     mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=False)

