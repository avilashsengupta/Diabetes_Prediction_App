import pandas as pd
from datetime import datetime as dt
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
# source = 'https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset'
app = Flask(__name__)

class DiabetesPrediction:
    def __init__(self):
        self.model = RandomForestClassifier(random_state = 0, n_estimators = 100)
        self.data = pd.read_csv('diabetes_prediction_dataset.csv')
    
    def label_encode(self, colname):
        trancol = LabelEncoder().fit_transform(self.data[colname].values)
        colloct = list(self.data.columns).index(colname)
        self.data = self.data.drop(colname, axis = 'columns')
        self.data.insert(loc = colloct, column = colname, value = trancol)
        return self.data
    
    def train_model(self):
        data = pd.read_csv('diabetes_prediction_dataset.csv')
        x = data.drop(['diabetes'], axis = 'columns')
        x = self.label_encode('gender')
        x = self.label_encode('smoking_history')
        x = x.drop(set(x.columns) - {'age','bmi','HbA1c_level','blood_glucose_level'}, axis = 'columns')
        y = data['diabetes'].values
        self.model.fit(x.iloc[:,:].values, y)

    def calculate_age(self, dob):
        y = int(dt.now().strftime('%Y')) - int(dob.split('-')[0])
        if int(dt.now().strftime('%m')) < int(dob.split('-')[1]): y = y - 1
        elif int(dt.now().strftime('%m')) == int(dob.split('-')[1]):
            if int(dt.now().strftime('%d')) < int(dob.split('-')[2]): y = y - 1
        return y

    def calculate_bmi(self, height, weight):
        bmi = weight / ((height / 100) * (height / 100))
        return round(bmi, 2)

    def predict_diabetes(self, dob, height, weight, hba1c, glucose):
        age = self.calculate_age(dob)
        bmi = self.calculate_bmi(height, weight)
        self.train_model()
        pred = self.model.predict([[age, bmi, hba1c, glucose]])
        return pred[0]

dbp = DiabetesPrediction()
@app.route('/', methods = ['GET', 'POST'])
def homepage():
    age = 0
    bmi = 0
    dbt = ' '
    if request.method == 'POST':
        dob = request.form['dob']
        height = request.form['height']
        weight = request.form['weight']
        hba1c = request.form['hba1c']
        glucose = request.form['glucose']
        age = dbp.calculate_age(dob)
        bmi = dbp.calculate_bmi(float(height), float(weight))
        dbt = dbp.predict_diabetes(dob, float(height), float(weight), float(hba1c), float(glucose))

    return render_template(
        template_name_or_list = 'home.html',
        age = age,
        bmi = bmi,
        dbt = dbt
    )
if __name__ == '__main__': app.run(debug = True)