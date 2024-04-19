import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
class DecisionTree:
    def __init__(self,data,x,y,max_depth=10):
        self.x=x
        self.y=y
        self.x_df=pd.DataFrame(data[x])
        self.y_df=pd.DataFrame(data[y])
        self.x_train,self.x_test,self.y_train,self.y_test= train_test_split(self.x_df, self.y_df, test_size = 0.25)
        self.xy_train= pd.concat([self.x_train, self.y_train], axis = 1).reindex(self.x_train.index)
        self.apply_tree(max_depth)

    def x_information(self):
        if type(self.x)!=str:
            return
        print("'"+self.x+"' describe:")
        print(self.x_train.describe())
        print()
    def y_information(self):
        print("'"+self.y+"' value counts:")
        print(self.y_train[self.y].describe())
        print()
    def x_diagram(self):
        if type(self.x)!=str:
            return
        f, axes = plt.subplots(1, 3, figsize=(24, 6))
        sb.boxplot(data = self.x_train, orient = "h", ax = axes[0])
        sb.histplot(data = self.x_train, ax = axes[1])
        sb.violinplot(data = self.x_train, orient = "h", ax = axes[2])
    def y_diagram(self,order=None):
        sb.catplot(y = self.y, data = self.y_train, kind = "count",order=order)
    def xy_diagram(self,order=None):
        if type(self.x)!=str:
            return
        f, axes = plt.subplots(1, 1, figsize=(18, 24))
        sb.violinplot(x = self.x, y =self.y, data =self.xy_train, orient = "h",order=order)
    def apply_tree(self,max_depth):
        self.tree = DecisionTreeClassifier(max_depth = max_depth)
        self.tree.fit(self.x_train,self.y_train)
        self.y_train_pred=self.tree.predict(self.x_train)
        self.y_test_pred=self.tree.predict(self.x_test)
    def apply_RandomForest(
            self,
            n_estimators=20,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=7,
            criterion='gini'
        ):        
        self.tree=RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
        )
        self.tree.fit(self.x_train,self.y_train)
        self.y_train_pred=self.tree.predict(self.x_train)
        self.y_test_pred=self.tree.predict(self.x_test)
    def draw_tree(self):
        f = plt.figure(figsize=(12,12))
        plot_tree(self.tree, filled=True, rounded=True, feature_names=self.x if type(self.x)!=str else [self.x])
    def print_goodness(self):
        if type(self.x)==str:
            print("'"+self.x+"':")
        print("train: ",round(self.tree.score(self.x_train, self.y_train),2))
        print("test: ",round(self.tree.score(self.x_test, self.y_test),2))
        print()
    def draw_matrix(self):
        f, axes = plt.subplots(2, 1, figsize=(12, 24))
        if type(self.x)==str:    
            print("'"+self.x+"' - train & test:")
        sb.heatmap(confusion_matrix(self.y_train, self.y_train_pred),annot = True, fmt=".0f", annot_kws={"size": 18}, ax = axes[0])
        sb.heatmap(confusion_matrix(self.y_test, self.y_test_pred), annot = True, fmt=".0f", annot_kws={"size": 18}, ax = axes[1])

