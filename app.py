from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('CleanedData.csv')
pipe = pickle.load(open("RidgeModel.pkl",'rb'))


@app.route('/')
def index():

    locations = sorted(data['site_location'].unique())
    return render_template('index.html', locations = locations)

@app.route('/predict', methods=['POST'])
def predict():
    locations = request.form.get('site_location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    input = pd.DataFrame([[locations, sqft, bath, bhk]],columns=['site_location','total_sqft','bath','bhk'])
    prediction = pipe.predict(input)[0] * 100000

    return str(prediction)

if __name__ == "__main__":
    app.run(debug=False, port=5001)
