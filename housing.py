import pandas as pd
import numpy as np
import sklearn 
import matplotlib as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

# data preprocessing 

housing_data = pd.read_csv('housing.csv')
features = ['bedrooms', 
            'bathrooms', 
            'area', 
            'parking', 
            'stories',
            'prefarea_yes',
            'furnishingstatus_furnished', 
            'furnishingstatus_semi-furnished', 
            'furnishingstatus_unfurnished', 
            'mainroad_no', 
            'mainroad_yes',
            'guestroom_no', 
            'guestroom_yes', 
            'basement_no', 
            'basement_yes', 
            'hotwaterheating_no', 
            'hotwaterheating_yes', 
            'airconditioning_no', 
            'airconditioning_yes', 
            'prefarea_no']

y = housing_data.price
#print(housing_data.columns)

# convert boolean values yes and no into integers 0 and 1
housing_data = pd.get_dummies(data = housing_data, dtype = 'int')
#print(housing_data)
#housing_data = housing_data.drop('price', axis=1, inplace=True)

#print(y)

#print(housing_data.head())
x = housing_data[features]
#print(x.head())

'''
# untrained lr model predicting the housing prices 
housing_model = LinearRegression()

housing_model = housing_model.fit(x,y)

prediction = housing_model.predict(x.head())

print(x.head())
print(prediction)
'''
x_array = np.array(x)

y_array = np.array(y)

#print(x_array)
#print(y_array)

# splitting data using sklearn
x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, train_size = 0.6) 
#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)

# train the decision tree model 
housing_model = LinearRegression()
# housing_model = DecisionTreeRegressor(max_depth = 10, min_samples_split = 5, min_samples_leaf = 10, ccp_alpha = 0)
#housing_model = RandomForestRegressor()

housing_model = housing_model.fit(x_train, y_train)

# print's the training accuracy 
#print(housing_model.score(x_train,y_train)) # this computes the r^2

y_pred = housing_model.predict(x_test)
total = 0
iterations = 10
for i in range(iterations):
    r_squared = r2_score(y_test, y_pred)
    total = total + r_squared
print(total/iterations)
#for i in range(10):
    #prediction = housing_model.predict(x_test)
    
#difference = abs(prediction - y_test)
#print(difference)
#print(y_test)
