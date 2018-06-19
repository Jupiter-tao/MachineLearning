from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

np.random.seed(0)              #设定随机种子
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75    #随机选择训练集
train, test = df[df['is_train']==True], df[df['is_train']==False]
features = df.columns[:4]

y = pd.factorize(train['species'])[0]                       #将不同类别命名为0，1，2
clf = RandomForestClassifier(n_jobs=2, random_state=0)      #创建一个随机森林分类器
clf.fit(train[features], y)                                 #训练分类器

clf.predict(test[features])                                 #预测
preds = iris.target_names[clf.predict(test[features])]      #将分类名称转换为实际的类名
pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species'])  #创建一个 confusion matrix
list(zip(train[features], clf.feature_importances_))        #查看特征重要性