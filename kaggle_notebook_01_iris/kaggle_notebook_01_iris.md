## Kaggle Notebook Analysis 01
- Date : 2021.07.14 20:00~ 
- Subject : Visualization and ML with the Iris Dataset
- Main Point : Introduction to Logistic Regression 
- Reference : 
    - [Python Data Visualizations](https://www.kaggle.com/benhamner/python-data-visualizations) form Kaggle
    - [Machine Learning with Iris Dataset](https://www.kaggle.com/jchen2186/machine-learning-with-iris-dataset) from Kaggle
    - [Logistic Regression](https://ko.wikipedia.org/wiki/%EB%A1%9C%EC%A7%80%EC%8A%A4%ED%8B%B1_%ED%9A%8C%EA%B7%80) from Wiki



```python
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# iris = pd.read_csv("../input/Iris.csv")
iris = sns.load_dataset("iris")

# iris.head()
iris.shape
```




    (150, 5)




```python
iris['species'].value_counts()
```




    setosa        50
    virginica     50
    versicolor    50
    Name: species, dtype: int64



## Seaborn Plot


```python
iris.plot(kind="scatter", x="sepal_length", y="sepal_width")
```

    *c* argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with *x* & *y*.  Please use the *color* keyword-argument or provide a 2-D array with a single row if you intend to specify the same RGB or RGBA value for all points.





    <AxesSubplot:xlabel='sepal_length', ylabel='sepal_width'>




    
![png](output_4_2.png)
    



```python
sns.jointplot(x="sepal_length", y="sepal_width", data=iris, size=5)

```




    <seaborn.axisgrid.JointGrid at 0x7fe2e8ef7370>




    
![png](output_5_1.png)
    



```python
ax = sns.FacetGrid(iris, hue="species", size=5)\
    .map(plt.scatter, "sepal_length", "sepal_width")\
    .add_legend()
ax = sns.jointplot(x="sepal_length", y="sepal_width", data=iris, size=5)

```


    
![png](output_6_0.png)
    



    
![png](output_6_1.png)
    



```python
sns.boxplot(x="species", y="petal_length", data=iris)
```




    <AxesSubplot:xlabel='species', ylabel='petal_length'>




    
![png](output_7_1.png)
    



```python
ax = sns.boxplot(x="species", y="petal_length", data=iris)
ax = sns.stripplot(x="species", y="petal_length", data=iris, jitter=True, edgecolor="gray")
```


    
![png](output_8_0.png)
    



```python
ax = sns.boxplot(x="species", y="petal_length", data=iris)
ax = sns.stripplot(x="species", y="petal_length", data=iris, jitter=True, edgecolor="gray")
```


    
![png](output_9_0.png)
    



```python
sns.violinplot(x="species", y="petal_length", data=iris, size=20)

```




    <AxesSubplot:xlabel='species', ylabel='petal_length'>




    
![png](output_10_1.png)
    



```python
sns.FacetGrid(iris, hue="species", size=6) \
   .map(sns.kdeplot, "petal_length") \
   .add_legend()
```




    <seaborn.axisgrid.FacetGrid at 0x7fe2c8f26e50>




    
![png](output_11_1.png)
    



```python
sns.pairplot(iris, hue="species", size=3, diag_kind='hist') # default='kde'(Kernel Density Estimator)

```




    <seaborn.axisgrid.PairGrid at 0x7fe2e9465700>




    
![png](output_12_1.png)
    


# pandas plot


```python
iris.boxplot(by="species", figsize=(12, 6))

```




    array([[<AxesSubplot:title={'center':'petal_length'}, xlabel='[species]'>,
            <AxesSubplot:title={'center':'petal_width'}, xlabel='[species]'>],
           [<AxesSubplot:title={'center':'sepal_length'}, xlabel='[species]'>,
            <AxesSubplot:title={'center':'sepal_width'}, xlabel='[species]'>]],
          dtype=object)




    
![png](output_14_1.png)
    



```python
from pandas.plotting import andrews_curves

plt.figure(figsize=(8,6))
andrews_curves(iris, "species")
```




    <AxesSubplot:>




    
![png](output_15_1.png)
    



```python
from pandas.plotting import parallel_coordinates
plt.figure(figsize=(8,6))
parallel_coordinates(iris, "species")
```




    <AxesSubplot:>




    
![png](output_16_1.png)
    



```python
from pandas.plotting import radviz
plt.figure(figsize=(8,6))
radviz(iris, "species")
```




    <AxesSubplot:>




    
![png](output_17_1.png)
    



```python
import plotly.express as px

hour = [ 1.  ,  2.15,  3.3 ,  4.45,  5.6 ,  6.75,  7.9 ,  9.05, 10.2 ,
       11.35, 12.5 , 13.65, 14.8 , 15.95, 17.1 , 18.25]



# px.scatter()
```


```python
import numpy as np
np.arange(1,19,1.15)
```




    array([ 1.  ,  2.15,  3.3 ,  4.45,  5.6 ,  6.75,  7.9 ,  9.05, 10.2 ,
           11.35, 12.5 , 13.65, 14.8 , 15.95, 17.1 , 18.25])



## Logistic Regression


```python
import numpy as np
import pandas as pd
import seaborn as sns
# sns.set_palette('husl')
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# data = pd.read_csv('../input/Iris.csv')
data = sns.load_dataset("iris")
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>virginica</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 5 columns</p>
</div>




```python
g = sns.pairplot(data, hue='species')
plt.show()
```


    
![png](output_22_0.png)
    



```python
g = sns.violinplot(y='species', x='sepal_length', data=data, inner='quartile')
plt.show()
g = sns.violinplot(y='species', x='sepal_width', data=data, inner='quartile')
plt.show()
g = sns.violinplot(y='species', x='petal_length', data=data, inner='quartile')
plt.show()
g = sns.violinplot(y='species', x='petal_width', data=data, inner='quartile')
plt.show()
```


    
![png](output_23_0.png)
    



    
![png](output_23_1.png)
    



    
![png](output_23_2.png)
    



    
![png](output_23_3.png)
    



```python
X = data.drop('species', axis=1)
y = data['species']
# print(X.head())
print(X.shape)
# print(y.head())
print(y.shape)
```

    (150, 4)
    (150,)


## 동일한 데이터 세트에서 train과 test 


```python
k_range = list(range(1,26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    y_pred = knn.predict(X)
    scores.append(metrics.accuracy_score(y, y_pred))
    
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()
```


    
![png](output_26_0.png)
    



```python
logreg = LogisticRegression()
logreg.fit(X, y)
y_pred = logreg.predict(X)
print(metrics.accuracy_score(y, y_pred))
```

    0.9733333333333334


## train과 test을 나누어서 

### 장점
- 두 데이터 셋에서 서로 다른 결과가 나올 것이다.
- 전체 데이터 셋을 사용하는 것보다 유동적이고 빠르다.(more flexible & faster)

### 단점
- 테스트마다 모델 성능 지표인 정확도가 달라질 수 있다. 
- 위의 단점을 k-fold 교차검증을 이용해서 해소할 수 있다.

### 주의
- 정확도는 random_state에 의존한다
- 모델이 복잡해질수록 훈련 데이터 셋의 정확도는 올라간다.
- 모델이 너무 과도하게 복잡하거나 또는 적당히 복잡하지 않으면(too complex or not complex enough), 테스트 셋의 정확도는 내려간다.
- KNN 모델의 경우, k 값이 복잡도를 결정한다. k가 낮을수록(lower value of k), 모델은 더 복잡해진다.(more complex) 


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```

    (90, 4)
    (90,)
    (60, 4)
    (60,)



```python
# experimenting with different n values
k_range = list(range(1,26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
    
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()
```


    
![png](output_30_0.png)
    



```python
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
```

    0.9833333333333333



```python
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X, y)

# make a prediction for an example of an out-of-sample observation
knn.predict([[6, 3, 4, 2]])
```




    array(['versicolor'], dtype=object)



![%E1%84%82%E1%85%A9%E1%84%90%E1%85%B3%E1%84%87%E1%85%AE%E1%86%A8.png](attachment:%E1%84%82%E1%85%A9%E1%84%90%E1%85%B3%E1%84%87%E1%85%AE%E1%86%A8.png)


```python

```
