import pandas as pd # data preprocessing
import numpy as np #mathematical computation
import matplotlib.pyplot as plt
import seaborn as sns

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

# To check for outliers.
print(df.describe())

# EDA analysis.
sns.countplot(y=df['fuel']) # Count for fuel type
print(plt.show())
sns.countplot(y=df['seller_type']) # Count for fuel type
print(plt.show())
sns.countplot(y=df['transmission']) # Count for fuel type
print(plt.show())
sns.countplot(y=df['owner']) # Count for fuel type
print(plt.show())

#Checking for Correlations amongs the columns.
corr = df.corr()
#corr = corr[corr>0.7] 
sns.heatmap(corr,annot=True,cmap='RdBu')
#plt.show()
print("==============================All the preprocessing steps are done===============================================")