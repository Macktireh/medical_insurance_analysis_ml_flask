import pandas as pd
import numpy as np
import flask
import joblib
from flask import request, render_template

app = flask.Flask(__name__)
#app.config["DEBUG"] = True

from flask_cors import CORS
CORS(app)

@app.route('/')
def home():
#    return "<h2 style='text-align:center;'> Bienvenue dans notre API (modèle ML en production). Utiliser '/predict' en POST pour faire des prédictions sur les charges médicaux </h1>"
    return render_template('home.html')

@app.route('/menu')
def menu():
    return render_template('index.html')


@app.route('/predict',methods=['GET'])
def predict():

    model = joblib.load('pipeline.pkl')

    cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    df = pd.DataFrame([[
                        int(request.args['age']), 
                        request.args['sex'], 
                        float(request.args['bmi']), 
                        int(request.args['children']), 
                        request.args['smoker'], 
                        request.args['region']
                    ]], columns=cols)
    
    charges_predict = model.predict(df)[0]
    return str(charges_predict)


if __name__ == "__main__":
    app.run(debug=True)