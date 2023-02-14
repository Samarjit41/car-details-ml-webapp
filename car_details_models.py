import pandas as pd # data preprocessing
import numpy as np #mathematical computation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split # to split data into tain and split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import pickle

#loading the data sheet and preprocessing
print("=================================loading the data sheet and preprocessing===============================")
df = pd.read_csv('C:\Capstone project\CAR DETAILS.csv')
print(df.shape) # rows=4340,cols=8
print(df.head())
print(df.columns)
# to check for null values and we found no values in this
print(df.isnull().sum()) 

# To check for duplicated values.
print("Duplicated values=======",df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("After dropping the duplicated values===",df.duplicated().sum())
print("Types=============",df.dtypes)
print("==============================All the preprocessing steps are done===============================================")

print("=================================Preparing For ML Model==========================================================")
# Selecting the dependent and independent features.
x = df.drop('selling_price',axis=1)
y = df['selling_price']
print(type(x))
print(type(y))
print(x.shape)
print(y.shape)

#Split the data into train and test.
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=2,random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#Evaluating the model  -R2_Score,MSE,RMSE,MAE

def eval_model(ytest,ypred):
    mae = mean_absolute_error(y_test,ypred)
    mse = mean_squared_error(ytest,ypred)
    rmse = np.sqrt(mean_squared_error(y_test,ypred))
    r2s = r2_score(y_test,ypred)
    print('MAE',mae)
    print('MSE',mse)
    print('RMSE',rmse)
    print('R2S',r2s)

print(x.columns)
print(x.head(2))

#Linear regression
step1 = ColumnTransformer(transformers=
                          [('col_transf',OneHotEncoder(drop='first',sparse = False),[0,3,4,5,6])],
                          remainder='passthrough')
print(step1)
step2 = LinearRegression()
pipe_lr = Pipeline([('step1',step1),('step2',step2)])
pipe_lr.fit(x_train,y_train)
ypred_lr = pipe_lr.predict(x_test)
eval_model(y_test,ypred_lr)
"""MAE 78586.39440207928
MSE 6915588975.508049
RMSE 83160.02029525996
R2S 0.5745889134635571"""

#Ridge Regression.
step1 = ColumnTransformer(transformers=
                          [('col_transf',OneHotEncoder(drop='first',sparse = False),[0,3,4,5,6])],
                          remainder='passthrough')
print(step1)
step2 = Ridge(alpha=10)
pipe_ridge = Pipeline([('step1',step1),('step2',step2)])
pipe_ridge.fit(x_train,y_train)
ypred_ridge = pipe_ridge.predict(x_test)
eval_model(y_test,ypred_ridge)
"""MAE 101622.54128474742
MSE 20525153338.36775
RMSE 143266.02297253787
R2S -0.26260074361354846"""

#Lasso Regression
step1 = ColumnTransformer(transformers=
                          [('col_transf',OneHotEncoder(drop='first',sparse = False),[0,3,4,5,6])],
                          remainder='passthrough')
print(step1)
step2 = Lasso(alpha=0.1)
pipe_Lasso = Pipeline([('step1',step1),('step2',step2)])
pipe_Lasso.fit(x_train,y_train)
ypred_Lasso = pipe_Lasso.predict(x_test)
eval_model(y_test,ypred_Lasso)
"""MAE 84129.68849280849
MSE 8174808695.130636
RMSE 90414.64867559148
R2S 0.4971282617374464"""

#KNN Regression.
step1 = ColumnTransformer(transformers=
                          [('col_transf',OneHotEncoder(drop='first',sparse = False),[0,3,4,5,6])],
                          remainder='passthrough')
print(step1)
step2 = KNeighborsRegressor(n_neighbors=10)
pipe_knn = Pipeline([('step1',step1),('step2',step2)])
pipe_knn.fit(x_train,y_train)
ypred_knn = pipe_knn.predict(x_test)
eval_model(y_test,ypred_knn)
"""MAE 160349.95
MSE 25759028280.005005
RMSE 160496.19397357997
R2S -0.5845615243371014"""

#DT Regression.
step1 = ColumnTransformer(transformers=
                          [('col_transf',OneHotEncoder(drop='first',sparse = False),[0,3,4,5,6])],
                          remainder='passthrough')
print(step1)
step2 = DecisionTreeRegressor(max_depth=30,min_samples_split=15)
pipe_dt = Pipeline([('step1',step1),('step2',step2)])
pipe_dt.fit(x_train,y_train)
ypred_dt = pipe_dt.predict(x_test)
eval_model(y_test,ypred_dt)
"""MAE 101805.79063360882
MSE 10600077874.32859
RMSE 102956.6796003474
R2S 0.34793830838424666"""

#Random Forest.
step1 = ColumnTransformer(transformers=
                          [('col_transf',OneHotEncoder(drop='first',sparse = False),[0,3,4,5,6])],
                          remainder='passthrough')
print(step1)
step2 = RandomForestRegressor(n_estimators=100,max_depth=50,min_samples_split=15,random_state=3)
pipe_rf = Pipeline([('step1',step1),('step2',step2)])
pipe_rf.fit(x_train,y_train)
ypred_rf = pipe_rf.predict(x_test)
eval_model(y_test,ypred_rf)
"""MAE 101671.25045406833
MSE 12982289800.579502
RMSE 113939.85167876734
R2S 0.20139701342071503"""
## Linear regression is the best model.

#Saving the model.
pickle.dump(pipe_lr,open('lrmodel.pkl','wb')) # Saving the model.
pickle.dump(df,open('data.pkl','wb')) #Saving the data frame






