import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from joblib import load
import importlib
app = Flask(__name__)
model = pickle.load(open('wind.pkl', 'rb'))

@app.route('/')
def h():
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
    
    
    x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0,TRAIN_SPLIT, past_history,future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0],TRAIN_SPLIT, None, past_history,future_target, STEP)