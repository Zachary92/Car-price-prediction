#data collection and cleaning
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

cars=pd.read_csv('C://Users/leoho/OneDrive/桌面/531/CarPrice.csv')

#splitting car brand and car model from CarName column
cars[['car_brand','car_model']]=\
    cars['CarName'].str.split(' ',1,expand=True)
cars.drop('CarName',inplace=True,axis=1)
#print(cars['car_brand'].unique())

#replace the incorrect spelling of the car brand
def replace_name(a,b):
    cars['car_brand'].replace(a,b,inplace=True)

replace_name('maxda','mazda')
replace_name('porcshce','porsche')
replace_name('toyouta','toyota')
replace_name('vokswagen','volkswagen')
replace_name('vw','volkswagen')
#print(cars['car_brand'].unique())