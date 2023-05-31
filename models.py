#model training and evaluation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from dv import cars
import warnings
warnings.filterwarnings('ignore')

car_features=['horsepower','enginesize','curbweight',
              'carwidth','carlength','highwaympg','citympg']
x=cars[car_features]
y=cars.price
train_x,test_x,train_y,test_y=train_test_split(x,y,random_state=0)

#linear model
Input=[('scaler',StandardScaler()),('mode',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(train_x,train_y)
linear=pipe.score(test_x,test_y)
linear_predict=pipe.predict(test_x)

#it's prediction
linear_predict_output=pd.DataFrame({'Horsepower':test_x.horsepower,'Enginesize':test_x.enginesize,
                                  'Curbweight':test_x.curbweight,'SalePrice':linear_predict})

#polynomial model using ridge
Input_poly=[('scaler',StandardScaler()),
            ('polynomial',PolynomialFeatures(degree=2,
            include_bias=False)),('model',Ridge(alpha=0.1))]
pipe_poly=Pipeline(Input_poly)
pipe_poly.fit(train_x,train_y)
polynomial=pipe_poly.score(test_x,test_y)
polynomial_predict=pipe_poly.predict(test_x)

#it's prediction
poly_predict_output=pd.DataFrame({'Horsepower':test_x.horsepower,'Enginesize':test_x.enginesize,
                                  'Curbweight':test_x.curbweight,'SalePrice':polynomial_predict})


car_brand2=pd.get_dummies(cars['car_brand'],drop_first=True)
car_model2=pd.get_dummies(cars['car_model'],drop_first=True)
fuel_type2=pd.get_dummies(cars['fueltype'],drop_first=True)
X=pd.concat([car_brand2,car_model2,
             fuel_type2,cars[['highwaympg','citympg']]],axis=1)
Y=cars.price
train_X,test_X,train_Y,test_Y=train_test_split(X,Y,random_state=0)

#decision tree model
Input1=[('scaler',StandardScaler()),
        ('model',DecisionTreeRegressor())]
pipe1=Pipeline(Input1)
pipe1.fit(train_X,train_Y)
decision_tree=pipe1.score(test_X,test_Y)
decision_tree_predict=pipe1.predict(test_X)

#it's prediction
dt_predict_output=pd.DataFrame({'Car_Brand':car_brand2.columns[test_X.iloc[:,0].values],
                                'Car_Model':car_model2.columns[test_X.iloc[:,0].values],
                                'Highway_mpg':test_X.iloc[:,-2],
                                'City_mpg':test_X.iloc[:,-1],
                                'Sales_Price':decision_tree_predict})

#random forest model 
Input2=[('scaler',StandardScaler()),
        ('model',RandomForestRegressor())]
pipe2=Pipeline(Input2)
pipe2.fit(train_X,train_Y)
random_forest=pipe2.score(test_X,test_Y)
random_forest_predict=pipe2.predict(test_X)

#it's prediction
rf_predict_output=pd.DataFrame({'Car_Brand':car_brand2.columns[test_X.iloc[:,0].values],
                                'Car_Model':car_model2.columns[test_X.iloc[:,0].values],
                                'Highway_mpg':test_X.iloc[:,-2],
                                'City_mpg':test_X.iloc[:,-1],
                                'Sales_Price':random_forest_predict})

#R squared score for each model for evaluating
rscore_results=pd.DataFrame({'Linear':linear,'Polynomial': polynomial,
                             'DecisionTree':decision_tree,
                             'RandomForest':random_forest},index=[0])
print(rscore_results)
print(poly_predict_output.head())
print(rf_predict_output.head())