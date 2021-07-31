# 0. importing librariees .
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# 1. importing dataset .
dataset = pd.read_csv('data/Position_Salaries.csv')

# 2. spliting the data to Features and Target .
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values
y = y.reshape(-1,1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
y = sc.fit_transform(y)

# 3. creating support vector regressor models .
from sklearn.svm import SVR
reg = SVR(kernel='rbf')
reg.fit(X,y)


# predicting 
y_pred = reg.predict(sc.transform(np.array([[6.5]])))

                     # 4. visualisation the SVR 

plt.scatter( X , y , color = 'red')
plt.plot(X , y_pred , color = 'blue')