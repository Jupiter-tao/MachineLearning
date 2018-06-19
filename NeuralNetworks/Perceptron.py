import pandas as pd  
import numpy as np  
from sklearn.cross_validation import train_test_split  
from sklearn.preprocessing import StandardScaler   
from sklearn.linear_model import Perceptron

column_names = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape',  
                    'Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']    
    #从互联网读取指定数据  
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',  
                       names = column_names)  
data = data.replace(to_replace = '?', value = np.nan)   #将?替换为标准缺失值表示  
data = data.dropna(how = 'any')                         #丢弃带有缺失值的数据  
      
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], test_size=0.25, random_state=33)  

ss = StandardScaler()  
X_train = ss.fit_transform(X_train)  
X_test = ss.transform(X_test)  

per = Perceptron()
per.fit(X_train, y_train) 
print(per.predict(X_test))
print('Accuracy of PER Classifier:', per.score(X_test, y_test))  