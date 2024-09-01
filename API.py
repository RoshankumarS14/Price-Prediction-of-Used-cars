import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
import warnings
from datetime import datetime
from flask import Flask, request, jsonify
warnings.filterwarnings("ignore")

app = Flask(__name__)

df=pd.read_csv("vehicle_details_new.csv")

df.drop(df[df["fleet_status"]=="ASSIGNED_TO_AUCTION"].index,inplace=True)
df.drop(columns=["expected_sale_price","min_bid_price","sale_date","fleet_status","remarks","vehicle_id"],inplace=True)
df.drop(df[df["sale_price"].isnull()].index,inplace=True)

df["years since manufacture"] = df["year"].apply(lambda a : 2024-a)
df["purchase_date"] = pd.to_datetime(df["purchase_date"])
df["days since purchase"] = (datetime.now() -  df["purchase_date"]).dt.days   
df.drop(columns=["year","purchase_date"],inplace=True)

df["grade"] = df["grade"].replace({'GRADE_0':8,'GRADE_1':7,'GRADE_2':6,'GRADE_3':5,'GRADE_4':4,'GRADE_5':3,'GRADE_6':2,'COMPLETE':1,'SALVAGE':0})
df["color"] = df["color"].replace({'Black     ':'Black','Blue      ':'Blue','Grey      ':'Grey','White     ':'White','Silver    ':'Silver','Silver\xa0\xa0\xa0\xa0':'Silver','Beige     ':'Beige','Baige     ':'Beige','Red       ':'Red','Gray':'Grey','37281':'Red','White\xa0\xa0\xa0\xa0\xa0':'White','Golden    ':'Golden','Lt. Golden':'Golden', 'Brown     ':'Brown'})

df_num = df.select_dtypes(include=np.number)
df_cat = df.select_dtypes(include=object)

encoder ={}
for col in df_cat.columns:
    encoder[col] = {}
    for category in df_cat[col].unique():
        encoder[col][category] = df_num[df_cat[col]==category]["sale_price"].mean()
    df_cat[col] = df_cat[col].map(encoder[col])

X_num = df_num.drop(columns=["sale_price"],axis=True)
X = pd.concat([X_num,df_cat],axis=1)
y = df_num["sale_price"]

xgbregressor = XGBRegressor(random_state=100,n_estimators=100).fit(X,y)

# Endpoint to predict car prices for a dataset
@app.route('/predict_cars_prices', methods=['POST'])
def predict_cars_prices():
    data = request.get_json()
    df = pd.DataFrame(data)

    df["years since manufacture"] = df["year"].apply(lambda a : datetime.now().year-a)
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    df["days since purchase"] = (datetime.now() -  df["purchase_date"]).dt.days   
    df.drop(columns=["year","purchase_date"],inplace=True)

    df["grade"] = df["grade"].replace({'GRADE_0':8,'GRADE_1':7,'GRADE_2':6,'GRADE_3':5,'GRADE_4':4,'GRADE_5':3,'GRADE_6':2,'COMPLETE':1,'SALVAGE':0})
    df["color"] = df["color"].replace({'Black     ':'Black','Blue      ':'Blue','Grey      ':'Grey','White     ':'White','Silver    ':'Silver','Silver\xa0\xa0\xa0\xa0':'Silver','Beige     ':'Beige','Baige     ':'Beige','Red       ':'Red','Gray':'Grey','37281':'Red','White\xa0\xa0\xa0\xa0\xa0':'White','Golden    ':'Golden','Lt. Golden':'Golden', 'Brown     ':'Brown'})

    df_num = df.select_dtypes(include=np.number)
    df_cat = df.select_dtypes(include=object)

    encoder ={}
    for col in df_cat.columns:
        encoder[col] = {}
        for category in df_cat[col].unique():
            encoder[col][category] = df_num[df_cat[col]==category]["sale_price"].mean()
        df_cat[col] = df_cat[col].map(encoder[col])

    X_num = df_num.drop(columns=["sale_price"],axis=True)
    X = pd.concat([X_num,df_cat],axis=1)

    try:
        X.drop(columns = ["expected_sale_price","min_bid_price","sale_date","fleet_status","remarks","vehicle_id"],inplace=True)
    except:
        print("done")

    # Specify exact columns in the correct order
    expected_features = [
        'kilometer', 'grade', 'net_book_value', 'purchase_price', 
        'years since manufacture', 'days since purchase', 'color', 
        'make_name', 'model_name'
    ]

    # Reorder columns to match training data
    X = X[expected_features]

    predictions =  xgbregressor.predict(X)
    return jsonify(predictions.tolist())

# Endpoint to predict a single car price
@app.route('/predict_car_price', methods=['POST'])
def predict():
    data = request.get_json()
    manufacture_year = data['manufacture_year']
    color = data['color']
    make_name = data['make_name']
    model_name = data['model_name']
    kilometer = data['kilometer']
    grade = data['grade']
    net_book_value = data['net_book_value']
    purchase_price = data['purchase_price']
    purchase_date = data['purchase_date']
    grade = {'GRADE_0':8,'GRADE_1':7,'GRADE_2':6,'GRADE_3':5,'GRADE_4':4,'GRADE_5':3,'GRADE_6':2,'COMPLETE':1,'SALVAGE':0}[grade]
    years_since_manufacture = datetime.now().year - manufacture_year
    days_since_purchase = (datetime.now() - datetime.strptime(purchase_date,"%Y-%m-%d")).days
    make_value = encoder["make_name"][make_name]
    color_replace = {'Black     ':'Black','Blue      ':'Blue','Grey      ':'Grey','White     ':'White','Silver    ':'Silver','Silver\xa0\xa0\xa0\xa0':'Silver','Beige     ':'Beige','Baige     ':'Beige','Red       ':'Red','Gray':'Grey','37281':'Red','White\xa0\xa0\xa0\xa0\xa0':'White','Golden    ':'Golden','Lt. Golden':'Golden', 'Brown     ':'Brown'}
    if color in color_replace.keys():
        color = color_replace[color]
    if color in encoder["color"].keys():
        color_value = encoder["color"][color]
    else:
        color_value = y.mean()
    if model_name in encoder["model_name"].keys():
        model_value = encoder["model_name"][model_name]
    else:
        model_value = y.mean()
    data = pd.DataFrame({'kilometer':kilometer,'grade':grade,'net_book_value':net_book_value,'purchase_price':purchase_price,
                         'years since manufacture':years_since_manufacture,'days since purchase':days_since_purchase,'color':color_value,
                         'make_name':make_value,'model_name':model_value},index=[0])
    prediction =  round(float(xgbregressor.predict(data)[0]),2)
    return jsonify({'predicted_price': prediction})

if __name__ == '__main__':
    app.run(debug=True)