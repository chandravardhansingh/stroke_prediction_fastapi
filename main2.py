from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

app = FastAPI()

model = pickle.load(open('model.pkl','rb'))

@app.get('/')
def home():
    return "Hello"

class Stroke(BaseModel):
    gender  : str
    age  : int
    hypertension  : int
    heart_disease  : int
    ever_married  : str
    work_type  : str
    Residence_type  : str
    avg_glucose_level : int
    bmi : int
    smoking_status  : str

@app.post('/predict')
def predict_stroke(data: Stroke):
    data = data.dict()
    gender  = data['gender']
    age  = data['age']
    hypertension  = data['hypertension']
    heart_disease  = data['heart_disease']
    ever_married  = data['ever_married']
    work_type  = data['work_type']
    Residence_type  = data['Residence_type']
    avg_glucose_level = data['avg_glucose_level']
    bmi = data['bmi']
    smoking_status  = data['smoking_status']
    
    features = [gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]
    data= pd.read_csv("./healthcare-dataset-stroke-data.csv")
    X = data.drop(['id','stroke'],axis=1)

    column_name = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status']
    # features =  ['Male', 2.0, 0, 0, 'No', 'children', 'Rural', 56.77, 20.9,'Unknown']
    test_row = pd.DataFrame([features],columns=column_name)


    from sklearn.preprocessing import OneHotEncoder
    categ_cols = ['gender','ever_married','work_type','Residence_type','smoking_status']
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoder.fit(X[categ_cols])

    new_test_data = pd.DataFrame(encoder.transform(test_row[categ_cols]), columns=encoder.get_feature_names())
    num_testdata = test_row.drop(categ_cols, axis=1)
    final_testdata = pd.concat([num_testdata, new_test_data], axis=1)
    final_testdata=final_testdata.astype(float)


    model = pickle.load(open('model.pkl','rb'))
    predicted = model.predict(final_testdata)
    predicted = int(predicted[0])

    if predicted == 0:
        return  "No Risk "
    else:
        return "RISK "