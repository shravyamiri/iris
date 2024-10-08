import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
iris = pd.read_csv('/content/Iris.csv')
iris
iris.shape
iris.describe()
iris.groupby('Species').mean()
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=iris)
plt.show()
sns.lineplot(data=iris.drop(['Species'], axis=1))
plt.show()
iris.plot.hist(subplots=True, layout=(3,3), figsize=(10, 10), bins=20)
plt.show()
g = sns.FacetGrid(iris, col='Species')
g = g.map(sns.kdeplot, 'SepalLengthCm')
sns.pairplot(iris)
iris.hist(color= 'mediumpurple' ,edgecolor='black',figsize=(10,10))
plt.show()
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
x = iris.drop('Species', axis=1)
y= iris.Species

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=5)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(x_train, y_train)

knn.score(x_test, y_test)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x, y)
y_pred = logreg.predict(x)
print(metrics.accuracy_score(y, y_pred))
from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
svm.fit(x_train, y_train)

svm.score(x_test, y_test)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)

dtree.score(x_test, y_test)
