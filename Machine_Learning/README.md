# Regression
## 1.Linear Regression (simple)

**Objective**: Find the best target function 

h(x) = θ<sub>0</sub> + θ<sub>1</sub>x

**Cost function**: SSE

**Learning objective**: minimize cost function

**Iterative method**: (Batch) Gradient descent

use learning rate and partial derivative of cost function

If &alpha; is too small, GD can be slow

If &alpha; is too large, GD can overreach the minimum or fail to converge

```
import numpy as np
from sklearn.linear_model import LinearRegression 
x = np.array([[0,2,3]]).T
y = np.array([1,1,4])

lin_reg = LinearRegression()
lin_reg.fit(x, y)
print(lin_reg.intercept_, " , ", lin_reg.coef_)
x_n = np.array([[7, 10]]).T
y_p = lin_reg.predict(x_n)
print(y_p)
```
## 2.Linear Regression (multiple)

**Objective**: Find the best target function 

h(x) = θ<sub>0</sub> + θ<sub>1</sub>x + ... + θ<sub>n</sub>x<sub>n</sub>

**Cost function**: SSE

**Features Scaling**: The goal is to be on s similar scale. This improves the performance and training stability of the model
- Standardization (Z-score)
- Normalization (min-max scale)

**Learning objective**: minimize cost function

**Iterative method**: Gradient descent
- Stochastic
- Mini-batch
- Batch

```
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np
import pandas as pd
# import os
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

data = pd.read_csv('https://raw.githubusercontent.com/ekaratnida/Applied-machine-learning/master/Week04-workshop-1/data.txt')

data['bathroom'] = np.random.randint(1,5,size = 47)

print(data.head())

x = data[['area', 'rooms', 'bathroom']]
print(x.head())

y = data[["price"]] 
print(y.head())

"""Train model"""
print("Train")
lin_reg = LinearRegression()
lin_reg.fit(x, y)
print(lin_reg.intercept_)
print(lin_reg.coef_)

"""Predict"""
print("Predict")
#X_test = np.array([[2000,6]])
x_test = pd.DataFrame(
    {
    "area":[2000,3000],
    "rooms":[6,3],
    "bathroom": [2,3]
    }
)

result = lin_reg.predict(x_test)
print(result)
```
![image](https://user-images.githubusercontent.com/85028821/216637476-530f3ad3-d8bc-4232-ab8c-a045614e434e.png)

## 3.Polynomial Regression

h(x) = θ<sub>0</sub> + θ<sub>1</sub>x + θ<sub>2</sub>x<sup>2</sup>

Change to Polynomial Features and use Linear Regression method
```
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X_train)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y_train)

# Visualizing the Polymonial Regression results
#print(np.sort(X_train,axis=None))
test = pol_reg.predict(poly_reg.fit_transform(X_train))
#print(np.sort(test,axis=None))
plt.scatter(X_train, y_train, color='red')
plt.scatter(X_test, y_test, color='yellow')
plt.plot(np.sort(X_train,axis=None), np.sort(test,axis=None), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
```
![image](https://user-images.githubusercontent.com/85028821/216638017-16fe03be-be95-4e6b-b1f7-50aa60ee0874.png)

## Overfitting
- Underfit: Train accuracy = low , Test accuracy = low
- Good fit: Train accuracy = high , Test accuracy = high
- Overfit: Train accuracy = high , Test accuracy = low

### Options
- Reduce number of features
- Regularization

## Regularization
### Additional term of Lambda in the cost function 
- Ridge regression: reduce parameters close to zero (but not zero)
- Lasso regression: reduce the number of features
- Elastic Net: mix of both Ridge and Lasso

Defualt Polynomial Regression
```
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[10, 15, 20, 25, 30]]).T
y = np.array([[10, 30, 50, 51, 52]]).T

poly_reg=PolynomialFeatures(degree=4)
#print("X ",X)

X_poly = poly_reg.fit_transform(X)
print("X_poly ", X_poly)

lin_reg4=LinearRegression()
lin_reg4.fit(X_poly,y)
print("Coef ", lin_reg4.coef_)
print("Intercept ", lin_reg4.intercept_)
print("R^2 ", lin_reg4.score(X_poly,y))

y_pred = lin_reg4.predict(X_poly)
print("MSE polynomial = ", mean_squared_error(y,y_pred))

plt.scatter(X,y)
plt.plot(X,y_pred,'g-',label="Polynomial regression")
plt.legend() 
plt.show()
```
![image](https://user-images.githubusercontent.com/85028821/221398852-392bec8f-88f3-4bd1-8706-832c60a80f3e.png)

Polynomial Regression Add Ridge
```
from sklearn.linear_model import Ridge

#poly_reg=PolynomialFeatures(degree=4)

#X_poly = poly_reg.fit_transform(X)

clf = Ridge(alpha=100) # lambda in lecture, alpha in sklearn
clf.fit(X_poly, y)

print("Coef ", clf.coef_)
print("Intercept ", clf.intercept_)
print("R^2 ", clf.score(X_poly,y))

y_pred = clf.predict(X_poly)
print("MSE polynomial = ", mean_squared_error(y,y_pred))

plt.scatter(X,y)
plt.plot(X,y_pred,'b-',label="Polynomial ridge regression")
plt.legend() 
plt.show()
```
![image](https://user-images.githubusercontent.com/85028821/221398861-ab19753e-290a-4f01-94b7-e3a292dbb5d9.png)

# Classification
## 1.Logistic Regression

**Objective**: Find the best target function (applied sigmoid function to a linear function)

h(x) = σ(θ<sup>T</sup>x)

**Cost function**: Cross-entropy loss (log loss)

**Learning objective**: minimize cost function

```
from sklearn.datasets import make_classification
import numpy as np
X, Y = make_classification(n_samples=10, n_features=4)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=123)
clf.fit(X,Y)
print('Y:', Y)
print('predict:',clf.predict(X))
print('predict_prob:',clf.predict_proba(X))

#easy evaluate
print('score:',clf.score(X, Y))
```
![image](https://user-images.githubusercontent.com/85028821/220159150-07ccdbd1-db8a-4ffe-8027-07ba7db2c554.png)

## Model Evaluation

### Machine learning diagnosis
- Get more training examples >> Fix overfit (high variance)
- Try smaller sets of features >> Fix overfit (high variance)
- Try getting additional features >> Fix underfit (high bias)
- Try adding polynomial features >> Fix underfit (high bias)
- Try decreasing λ >> Fix underfit (high bias)
- Try increasing λ >> Fix overfit (high variance)

### Methods to create model
- **Hold-out**
```
import numpy as np
from numpy import random
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

X, y = np.arange(10).reshape((5, 2)),random.rand(5) # 4 columns, 100 rows, 80:20%
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
print('X train ', X_train)
print('y train ', y_train)
print('X test ', X_test)
print('y test ', y_test)

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model = lr.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('y_pred ', y_pred)
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test,y_pred))
```
- **Cross validation** (Grid search CV)

```
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
iris = datasets.load_iris()

# Create regularization penalty space
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)
print(C)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

clf = LogisticRegression()
model = GridSearchCV(clf, hyperparameters, cv=5, verbose=0)
best_model = model.fit(iris.data, iris.target)

# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
```

### Error metrics
- Confusion Matrix
- Precision, Recell, Accuracy
- F1-score
- ROC, AUC

**Confusion Matrix**
```
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

X, y = make_classification(random_state=0)
print('X = ',X)
print('y = ',y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) #75:25

clf = LogisticRegression()

clf.fit(X_train, y_train)

plot_confusion_matrix(clf, X_test, y_test)

plt.show()

## [TN,FP]
## [FN,TP]
```
![image](https://user-images.githubusercontent.com/85028821/221399726-3077f36f-4d8d-4e85-b416-0d727421af18.png)

```
from sklearn.metrics import confusion_matrix
y_true = [0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
y_pred = [0, 0, 0, 1, 0, 1, 0, 1, 1, 0]

test = confusion_matrix(y_true, y_pred)
print(test) # 2D
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() ## มันจะเรียง output แบบนี้ / ravel ทำจาก 2 มิติให้กลายเป็นเรียง 1 มิติ
print("tp=",tp," fp=",fp)
print("fn=",fn," tn=", tn)

from sklearn import metrics
from sklearn.metrics import accuracy_score
precision = metrics.precision_score(y_true, y_pred)
recall = metrics.recall_score(y_true, y_pred)
f1 = metrics.f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
print("accuracy = ",accuracy )
print("precision = ",precision)
print("recall = ", recall)
print("f1 = ", f1)

print('******************')

target_names = ['class 0', 'class 1']  ## ปกติจะดูที่ class 1 บอกว่ามี 1 อยู่ 6 ตัว (class 0 เหมือนเรามอง 0 เป็นตัวหลักแทน มันจะเหมือน invert กับclass 1 แต่ถ้าเหมือนที่เราเรียนมันจะดูใน class1)
print(metrics.classification_report(y_true, y_pred, target_names=target_names))

## macro คือเอาบวกกันหาร 2 แต่ w avg จะมีการเอาค่าคูณ support / all support
```
![image](https://user-images.githubusercontent.com/85028821/221399748-d60c60af-84df-4415-9c5a-22465f6c34bf.png)

**ROC AUC**
```
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
print(X.shape)

# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, random_state=2)

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(testy))]

# fit a model
model = LogisticRegression()
model.fit(trainX, trainy)

# predict probabilities
lr_probs = model.predict_proba(testX)

# keep probabilities for the positive outcome only
## เวลาค่าที่ออกมามันจะมี 2 column , columnแรกเป็นลบ columnสองเป็นบวก
lr_probs = lr_probs[:, 1]

# calculate scores
ns_auc = roc_auc_score(testy, ns_probs)
lr_auc = roc_auc_score(testy, lr_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()

# show the plot
pyplot.show()
```
![image](https://user-images.githubusercontent.com/85028821/221399786-ace82fad-ace3-440c-8b5c-c3e9fc873583.png)

## 2.K-Nearest Neighbors (K-NN)
Assumption: Similar Inputs have similar outputs

To classify a new input vector x, examine the k-closet trainging data points to x and assign the object to the most frequently occurring class

How to choose K: Larger K may lead to better performance use Rule of thumb ( k < sqrt(n) )

**Issues and Remedies**
- Ties
  - for binary classification: choose K odd
  - for multi-class classification: decrease K until the tie is broken 
- Attributes have larger ranges: Normalize scale
- Irrelevant/ correlated attributes: eliminate some attributes
- Expensive at test time: use subset of dimensions (features selection)

```
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

dataset = np.array([[1,2,0],
                    [1,2.5,0],
                    [7,2,1],
                    [3,2.3,1],
                    [4,2.1,1]])

test = np.array([[2,2]])

from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=3)
neigh.fit(dataset[:, :2], dataset[:,2])

result = neigh.kneighbors(test)
print("result = ",result) #distance and index

#Plot all points
plt.scatter(dataset[:, 0], dataset[:, 1],c = dataset[:,2], s=30, cmap='viridis')
#Plot neighbors
plt.scatter(dataset[result[1][0:], 0], dataset[result[1][0:], 1], c='red', s=200, alpha=0.7)
#Plot target
plt.scatter(test[0,0], test[0,1], c='green', s=100, alpha=0.7)
plt.show()
```
![image](https://user-images.githubusercontent.com/85028821/221400854-bbe463f3-7e86-45b4-bd6a-1236deb114cf.png)

KNeighborsClassifier
```
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import classification_report
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(dataset[:, :2], dataset[:,2])
answer = knn.predict(test)
print(answer)
```
> [1.]

## 3.Decision Tree
- **Classification And Regression Tree (CART)** : use Gini index
- **Iterative Dichotomiser 3 (ID3)** : use information gain (entropy)

### Issue: Overfitting
**Solution**
- Limit the number of iterations of ID3
- Pruning
- Random forests 

```
# perform training 
from sklearn.tree import DecisionTreeClassifier  # import the classifier
classifier = DecisionTreeClassifier(criterion='gini', max_depth=3)
classifier.fit(X, y)

#!pip install pip install pydotplus
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data, filled=True, rounded=True, special_characters=True,feature_names = feature_cols ,class_names=['no','yes'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('playtennis.png')
Image(graph.create_png())
```
![image](https://user-images.githubusercontent.com/85028821/221401097-e6a7003b-6984-46ef-9ed6-a9e36a8302fe.png)

## 4.Neural Network
### NN model layer
- Input layer
- Hidden layer: Linear operation + Activation function (eg. sigmoid)
- Output layer
### NN trainable parameter
- Weight
- Bias
### Gradient computation
**Backpropagation algorithm**

Example tensorflow code can see in https://github.com/macgyver121/Project-Predict-car-prices-with-ML-and-MLP
