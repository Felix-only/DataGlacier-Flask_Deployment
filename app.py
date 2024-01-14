import numpy as np
import pandas as pd
from flask import Flask, request,render_template, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='House price should be $ {}'.format(output))

@app.route('/api/')
def api():
    bed_room = request.args.get('bed_room')
    area = request.args.get('area')
    house_age = request.args.get('house_age')

    df = pd.DataFrame({'bed_room':[bed_room], 'area':[area], 'house_age':[house_age]})

    pred_price = model.predict(df)

    return jsonify({'House Price': str(pred_price)})

if __name__ == "__main__":
    app.run(debug=True)