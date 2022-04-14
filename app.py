
from Utils.conf import path_pipeline_obj,threshold
import pandas as pd
from Utils.model import PipelinePredictor
from flask import Flask, request, jsonify
import numpy as np


# App config.
app = Flask(__name__)

@app.route('/api/', methods=['POST'])
def makecalc():

    r = request.get_json()

    # Flatten data
    df = pd.DataFrame.from_dict(r)


    prediction,proba = model.predict(df)

    print("prediction :",prediction," proba :",proba)

    if proba[0][1] <= threshold:
        pred = 0
    else :
        pred = 1
    
    dict_final = {
        'prediction' : int(pred),
        'probabilite' : float(proba[0][1])
        }

    print('Nouvelle Prédiction : \n', dict_final)

    return jsonify(dict_final)

if __name__ == '__main__':
    model = PipelinePredictor(path_pipeline_obj)
    threshold = threshold
    app.run(debug=True)


