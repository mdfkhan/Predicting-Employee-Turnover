#Task 1: Import Libraries
from __future__ import print_function
#%matplotlib inline
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import pandas as pd
import pandas_profiling
plt.style.use("ggplot")
warnings.simplefilter("ignore")
plt.rcParams['figure.figsize'] = (12,8)

#Task 2: Exploratory Data Analysis

hr=pd.read_csv('employee_data.csv')
hr.head()
hr.profile_report(title='Data Report')
pd.crosstab(hr.salary,hr.quit).plot(kind='bar')
plt.title("Turnover Frequency on Salary Bracket")
plt.xlabel('Salary')
plt.ylabel('Frequency of Turnover')
plt.show()
pd.crosstab(hr.department,hr.quit).plot(kind='bar')
plt.title("Turnover Frequency on Department Bracket")
plt.xlabel('Department')
plt.ylabel('Frequency of Turnover')
plt.show()

#Task 3: Encode Categorical Features

cat_vars=['department','salary']
for var in cat_vars:
    cat_list=pd.get_dummies(hr[var],prefix=var)
    hr=hr.join(cat_list)
hr.head()
hr.drop(columns=['department','salary'],axis=1,inplace=True)
hr.head()

#Task 4: Visualize Class Imbalance

from yellowbrick.target import ClassBalance
plt.style.use("ggplot")
plt.rcParams['figure.figsize'] = (12,8)
visualizer = ClassBalance(label=['satyed','quit']).fit(hr.quit)
visualizer.show()

#Task 5: Create Training and Test Sets

X=hr.loc[:,hr.columns!='quit']
y=hr.quit
from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, Y, stratify=Y)
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2,stratify=y)

#Task 6 & 7: Build an Interactive Decision Tree Classifier

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz # display the tree within a Jupyter notebook
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from ipywidgets import interactive, IntSlider, FloatSlider, interact
import ipywidgets
from IPython.display import Image
from subprocess import call
import matplotlib.image as mpimg

@interact
def plot_tree(crit=['gini','entropy'],
              split=['best','random'],
              depth=IntSlider(min=1,max=30,value=2, continuous_update=False),
              min_split=IntSlider(min=2,max=5,value=2, continuous_update=False),
              min_leaf=IntSlider(min=1,max=5,value=1, continuous_update=False)):
    estimator=DecisionTreeClassifier(random_state=0,
                                     criterion=crit,
                                     splitter=split,
                                    max_depth=depth,
                                     min_samples_split=min_split,
                                     min_samples_leaf=min_leaf)
    estimator.fit(X_train,y_train)
    print('Decision Tree Training Accuracy:{:.3f}'.format(accuracy_score(y_train,
                                                        estimator.predict(X_train))))
    print('Decision Tree Test Accuracy:{:.3f}'.format(accuracy_score(y_test,
                                                                    estimator.predict(X_test))))
    graph = Source(tree.export_graphviz(estimator,out_file=None,
                                      feature_names=X_train.columns,
                                     class_names=['stayed','quit'],
                                       filled=True))
          
    display(Image(data=graph.pipe(format='png')))
    
    return(estimator)

#Task 8: Build an Interactive Random Forest Classifier

@interact
def plot_tree_rf(crit=['gini','entropy'],
                 bootstrap=['True','False'],
                 depth=IntSlider(min=1,max=30,value=2, continuous_update=False),
                 forests=IntSlider(min=1,max=200,value=100,continuous_update=False),
                 min_split=IntSlider(min=2,max=5,value=2, continuous_update=False),
                 min_leaf=IntSlider(min=1,max=5,value=1, continuous_update=False)):
    estimator=RandomForestClassifier(random_state=1,
                                    criterion=crit,
                                    bootstrap=bootstrap,
                                    n_estimators=forests,
                                    max_depth=depth,
                                    min_samples_split=min_split,
                                    min_samples_leaf=min_leaf,
                                    n_jobs=-1,
                                    verbose=False)
    estimator.fit(X_train,y_train)
    print('Random Forest Training Accuracy:{:.3f}'.format(accuracy_score(y_train,estimator.predict(X_train))))
    print('Random Forest Test Accuracy:{:.3f}'.format(accuracy_score(y_test,estimator.predict(X_test))))
    num_tree = estimator.estimators_[0]
    print('\Visualizing tree:',0)
    graph = Source(tree.export_graphviz(num_tree,
                                    out_file=None,
                                      feature_names=X_train.columns,
                                     class_names=['stayed','quit'],
                                       filled=True))
          
    display(Image(data=graph.pipe(format='png')))
    
    return(estimator)

#Task 9: Feature Importance and Evaluation Metrics


from yellowbrick.model_selection import FeatureImportances
plt.rcParams['figure.figsize'] = (12,8)
plt.style.use("ggplot")
rf=RandomForestClassifier(bootstrap='True', class_weight=None, criterion='gini',
            max_depth=5, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
            oob_score=False, random_state=1, verbose=False,
            warm_start=False)
viz=FeatureImportances(rf)
viz.fit(X_train,y_train)
viz.show();
dt=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=0,
            splitter='best')
viz=FeatureImportances(dt)
viz.fit(X_train,y_train)
viz.show();
from yellowbrick.classifier import ROCAUC
visualizer=ROCUAC(rf,classes=['stayed','quit'])
visualizer.fit(X_train,y_train)
visualizer.score(X_test,y_test)
