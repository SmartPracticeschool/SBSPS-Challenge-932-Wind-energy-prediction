import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from joblib import load
import importlib
app = Flask(__name__)
model = pickle.load(open('app.pkl', 'rb'))

@app.route('/')
def Webpage():
    return render_template("Webpage.html")

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[float(x) for x in request.form.values()]]
    print(x_test)
    sc = load('scalar2.save') 
    prediction = model.predict(sc.transform(x_test))
    print(prediction)
    
    return render_template("Webpage.html",prediction_text='Energy {}'.format(float(prediction)))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=False)
