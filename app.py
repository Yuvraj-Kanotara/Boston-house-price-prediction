import pickle
from flask import Flask,render_template,request,redirect,app,jsonify

import numpy as np
import pandas as pd

app = Flask(__name__)
##load the model
regpredict= pickle.load(open("regpredict.pkl","rb"))
scaler= pickle.load(open("scaler.pkl","rb"))
@app.route('/')
def home():
    return render_template("1.html")

@app.route("/predict_api",methods= ["POST"])

def predict_api():
    data= request.json["data"]
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output= regpredict.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route("/predict",methods= ["POST"])

def predict():
    data = [float(x) for x in request.form.values()]
    final_input= scaler.transform(np.array(data).reshape(1,-1))
    output= regpredict.predict(final_input)[0]
    return render_template('1.html', prediction_text= "The house price is {}".format(output))

if __name__ =="__main__":
    app.run(debug=True)