from flask import Flask,render_template,request,redirect
import joblib
import pandas as pd
import numpy as np

app=Flask(__name__)
model = joblib.load('car_price_grad_boost_model.pkl')
car=pd.read_csv('Cleaned_Car_data.csv')

companies=sorted(car['company'].unique())
car_models=sorted(car['name'].unique())
fuel_types=car['fuel_type'].unique()

@app.route('/',methods=['GET','POST'])
def index():
    prediction = None;
    return render_template('index.html',prediction=prediction, companies=companies, car_models=car_models,fuel_types=fuel_types)


@app.route('/predict', methods=['POST'])
def predict():
    car_company = request.form.get('company')
    car_model = request.form.get('car_models')
    car_year = request.form.get('year')
    car_fuel_type = request.form.get('fuel_type')
    car_driven = request.form.get('kilo_driven')

    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([car_model, car_company, car_year, car_driven, car_fuel_type]).reshape(1, 5)))
    
    prediction = str(np.round(prediction[0], -3))
    
    return render_template('index.html', 
                           prediction=prediction, 
                           companies=companies, 
                           car_models=car_models, 
                           fuel_types=fuel_types,
                           selected_company=car_company,
                           selected_model=car_model,
                           selected_year=car_year,
                           selected_kilo_driven=car_driven,
                           selected_fuel_type=car_fuel_type)



if __name__=='__main__':
    app.run()
