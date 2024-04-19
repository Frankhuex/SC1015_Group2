import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
sb.set_theme(font_scale=1)
class Classification:
    def __init__(self,data,x,y,max_depth=10):
        self.x=x
        self.y=y
        self.x_df=pd.DataFrame(data[x])
        self.y_df=pd.DataFrame(data[y])
        self.x_train,self.x_test,self.y_train,self.y_test= train_test_split(self.x_df, self.y_df, test_size = 0.25)
        self.xy_train= pd.concat([self.x_train, self.y_train], axis = 1).reindex(self.x_train.index)



    # Print information
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



    # Draw diagram
    def x_diagram(self,numerical=True):
        if type(self.x)!=str:
            return
        if numerical:
            f, axes = plt.subplots(1, 3, figsize=(24, 6))
            sb.boxplot(data = self.x_train, orient = "h", ax = axes[0])
            sb.histplot(data = self.x_train, ax = axes[1])
            sb.violinplot(data = self.x_train, orient = "h", ax = axes[2])
        else:
            sb.catplot(y = self.x, data = self.x_train, kind = "count")       
    
    def y_diagram(self,order=None):
        sb.catplot(y = self.y, data = self.y_train, kind = "count",order=order)
    
    def xy_diagram(self,order=None,numerical=True):
        if type(self.x)!=str:
            return
        if numerical:    
            f, axes = plt.subplots(1, 1, figsize=(18, 24))
            sb.violinplot(x = self.x, y =self.y, data =self.xy_train, orient = "h",order=order)
        else:
            self.cross_tab=pd.crosstab(self.xy_train[self.x],self.xy_train[self.y])
            plt.figure(figsize=(8, 6))
            sb.heatmap(self.cross_tab, annot = True, fmt=".0f", annot_kws={"size": 18})
            plt.xlabel(self.y)
            plt.ylabel(self.x)
            plt.title(f'Relationship between {self.x} & {self.y}')
            plt.show()



    # Apply algorithms
    def apply_tree(self,max_depth):
        self.tree = DecisionTreeClassifier(max_depth = max_depth)
        self.tree.fit(self.x_train,self.y_train)
        self.y_train_pred=self.tree.predict(self.x_train)
        self.y_test_pred=self.tree.predict(self.x_test)

    def apply_RandomForest(self,n_estimators=20,max_depth=10):      
        # min_samples_split=2,min_samples_leaf=1,max_features=7,criterion='gini'  
        self.tree=RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth
        )
        '''
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        '''
        self.tree.fit(self.x_train,self.y_train)
        self.y_train_pred=self.tree.predict(self.x_train)
        self.y_test_pred=self.tree.predict(self.x_test)
    
    def apply_more_models(self,models):
        self.model_accuracy={}
        self.best_model = None
        self.best_accuracy = 0
        for name, model in models:
            scores = cross_val_score(model, self.x_df, self.y_df, cv=5, scoring='accuracy')
            mean_accuracy = scores.mean()
            #print(f"{name} Mean Accuracy: {mean_accuracy:.4f}")
            self.model_accuracy[name]=mean_accuracy
            if mean_accuracy > self.best_accuracy:
                self.best_accuracy = mean_accuracy
                self.best_model = model

    

    # Result analysis
    def draw_tree(self):
        f = plt.figure(figsize=(12,12))
        plot_tree(self.tree, filled=True, rounded=True, feature_names=self.x if type(self.x)!=str else [self.x])
    
    def print_goodness(self):
        if type(self.x)==str:
            print("'"+self.x+"':")
        print("train: ",round(self.tree.score(self.x_train, self.y_train)*100,1),"%")
        print("test: ",round(self.tree.score(self.x_test, self.y_test)*100,1),"%")
        print()

    def print_more_models_result(self):
        for key in self.model_accuracy:
            print(key+":",round(self.model_accuracy[key]*100,1),"%")
    
    def draw_matrix(self):
        f, axes = plt.subplots(2, 1, figsize=(12, 24))
        if type(self.x)==str:    
            print("'"+self.x+"' - train & test:")
            plt.title(f"{self.x}'s Train & Test Confusion Matrix")
        else:
            print("Train & Test Confusion Matrix")
            plt.title("Train & Test Confusion Matrix")
        sb.heatmap(confusion_matrix(self.y_train, self.y_train_pred),annot = True, fmt=".0f", annot_kws={"size": 18}, ax = axes[0])
        sb.heatmap(confusion_matrix(self.y_test, self.y_test_pred), annot = True, fmt=".0f", annot_kws={"size": 18}, ax = axes[1])
        plt.show()

