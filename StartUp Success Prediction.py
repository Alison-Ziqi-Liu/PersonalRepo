# -*- coding: utf-8 -*-
"""

***All reasonings are discussed in the pdf***
***Thanks! Good day!***

"""

import pandas
import numpy as np

df_starter = pandas.read_excel("Kickstarter.xlsx")

#data preprocessing
df_starter = df_starter.loc[ (df_starter['state'] == 'failed') | (df_starter['state'] == 'successful') ]

    #delete invalid variables
df_starter = df_starter.drop(columns = ['project_id','name','pledged'])
df_starter = df_starter.drop(columns = ['staff_pick','spotlight','backers_count'])
df_starter = df_starter.drop(columns = ['state_changed_at','state_changed_at_weekday','state_changed_at_month','state_changed_at_day','state_changed_at_yr','state_changed_at_hr','launch_to_state_change_days'])

    #delete duplicates
df_starter.drop_duplicates()

    #manual feature selection & transformation 
df_starter['disable_communication'] = pandas.get_dummies(df_starter['disable_communication']) #col2

df_starter['bad_category'] = np.where((df_starter['category'] == 'Musical')|(df_starter['category'] == 'Plays'), True, False)
df_starter['bad_category'] = pandas.get_dummies(df_starter['bad_category']) #col32

df_starter['created_on_Sunday'] = np.where(df_starter['created_at_weekday']== 'Sunday', True, False)
df_starter['created_on_Sunday'] = pandas.get_dummies(df_starter['created_on_Sunday']) #col33

df_starter['launched_on_weekend'] = np.where((df_starter['launched_at_weekday']== 'Sunday')|(df_starter['launched_at_weekday']== 'Saturday')|(df_starter['launched_at_weekday']== 'Friday'), True, False)
df_starter['launched_on_weekend'] = pandas.get_dummies(df_starter['launched_on_weekend']) #col34

df_starter['deadline_yr_12-16'] = np.where((df_starter['deadline_yr']<= 2016)&(df_starter['deadline_yr']>= 2012), True, False)
df_starter['deadline_yr_12-16'] = pandas.get_dummies(df_starter['deadline_yr_12-16']) #col35

df_starter['created_yr_12-16'] = np.where((df_starter['created_at_yr']<= 2016)&(df_starter['created_at_yr']>= 2012), True, False)
df_starter['created_yr_12-16'] = pandas.get_dummies(df_starter['created_yr_12-16']) #col36

df_starter['launch_to_deadline_over60'] = np.where(df_starter['launch_to_deadline_days']>60, True, False)
df_starter['launch_to_deadline_over60'] = pandas.get_dummies(df_starter['launch_to_deadline_over60']) #col37

df_starter['create_to_launch_days_long'] = np.where(df_starter['create_to_launch_days']>1000, True, False)
df_starter['create_to_launch_days_long'] = pandas.get_dummies(df_starter['create_to_launch_days_long']) #col38

df_starter['create_to_launch_days_short'] = np.where(df_starter['create_to_launch_days']<500, True, False)
df_starter['create_to_launch_days_short'] = pandas.get_dummies(df_starter['create_to_launch_days_short']) #col39


X = df_starter.iloc[:,[0,2,8,11,12,13,14,32,33,34,35,36,37,38,39]]
y = df_starter["usd_pledged"]

X.info()
    #split for performance test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 5)
from sklearn.metrics import mean_squared_error

    #standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
X_train_std = scaler.fit_transform(X_train)


 ######################################### A. Regression
    #simple sklearn
from sklearn.linear_model import LinearRegression
model_sk_lr = LinearRegression() 
model_sk_lr.fit(X_train_std,y_train)
y_test_pred = model_sk_lr.predict(X_test)
print(mean_squared_error(y_test, y_test_pred))

     #Ridge
from sklearn.linear_model import Ridge
for i in range (500,5500,500):
    model_ridge = Ridge(alpha=i)
    model_ridge.fit(X_train_std,y_train)
    y_test_pred = model_ridge.predict(X_test)
    print('Alpha = ',i, 'MSE =',mean_squared_error(y_test, y_test_pred))

     #LASSO
from sklearn.linear_model import Lasso
for i in range (500,5500,500):
    model_LASSO = Lasso(alpha=i) 
    model_LASSO.fit(X_train_std,y_train) 
    y_test_pred = model_LASSO.predict(X_test)
    print('Alpha = ',i,' / MSE =',mean_squared_error(y_test, y_test_pred))

model_LASSO_opt = Lasso(alpha=5000) 
model_LASSO_opt.fit(X_train_std,y_train) 
model_LASSO_opt.coef_
pandas.DataFrame(list(zip(X.columns,model_LASSO_opt.coef_)), columns = ['predictor','coefficient'])

 ######################################### B.Classification  
X2 = df_starter.iloc[:,[0,2,8,11,12,13,14,32,33,34,35,36,37,38,39]]
y2 = df_starter["state"]

#split for performance test
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size = 0.33, random_state = 5)
from sklearn import metrics
    #standardize
scaler = StandardScaler() 
X_train_std2 = scaler.fit_transform(X_train2)

#sklearn logistic
from sklearn.linear_model import LogisticRegression
model_sk_lo = LogisticRegression(max_iter = 5000)
model_sk_lo.fit(X_train2, y_train2)

y_test_pred = model_sk_lo.predict(X_test2)
metrics.accuracy_score(y_test2, y_test_pred)
metrics.confusion_matrix(y_test2, y_test_pred)

#feature selection
    #Random Forest 
from sklearn.ensemble import RandomForestClassifier
model_randomforest = RandomForestClassifier(random_state=5) 
model_randomforest.fit(X_train2, y_train2) 
model_randomforest.feature_importances_
pandas.DataFrame(list(zip(X.columns,model_randomforest.feature_importances_)), columns = ['predictor','feature importance'])

from sklearn.model_selection import cross_val_score 
for i in range (2,7):
    model2 = RandomForestClassifier(random_state=5,max_features=i,n_estimators=100) 
    scores = cross_val_score(estimator=model2, X=X2, y=y2, cv=5) 
    print(i,':',np.average(scores))

        ###sklearn test
X2_selected = X.iloc[:,[0,2,3,4,5,6]]
X_train2_selected, X_test2_selected, y_train2_selected, y_test2_selected = train_test_split(X2_selected, y2, test_size = 0.33, random_state = 5)

model_sk_lo2 = LogisticRegression(max_iter = 5000)
model_sk_lo2.fit(X_train2_selected, y_train2_selected)
y_test_pred = model_sk_lo2.predict(X_test2_selected)
metrics.accuracy_score(y_test2_selected, y_test_pred)

    #RFE
from sklearn.feature_selection import RFE
lr = LogisticRegression(max_iter=5000) 
model_rfe = RFE(lr, n_features_to_select=4) 
model_rfe.fit(X_train2, y_train2)
model_rfe.ranking_
pandas.DataFrame(list(zip(X.columns,model_rfe.ranking_)), columns = ['predictor','ranking'])

y_test_pred = model_rfe.predict(X_test2)
metrics.accuracy_score(y_test2, y_test_pred)
metrics.confusion_matrix(y_test2, y_test_pred)

 ###sklearn test
X2_selected = X.iloc[:,[1,2,4,9]]
X_train2_selected, X_test2_selected, y_train2_selected, y_test2_selected = train_test_split(X2_selected, y2, test_size = 0.33, random_state = 5)

model_sk_lo2 = LogisticRegression(max_iter = 5000)
model_sk_lo2.fit(X_train2_selected, y_train2_selected)
y_test_pred = model_sk_lo2.predict(X_test2_selected)
metrics.accuracy_score(y_test2_selected, y_test_pred)


    #KNN
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=3,p=2) 
model_knn.fit(X_train_std2,y_train2)

y_test_pred = model_knn.predict(X_test2)
metrics.accuracy_score(y_test2, y_test_pred)
metrics.confusion_matrix(y_test2, y_test_pred)

for i in range (1111,1120):
    model_knn = KNeighborsClassifier(n_neighbors=i) 
    model_knn.fit(X_train_std2,y_train2) 
    y_test_pred = model_knn.predict(X_test2) 
    print(metrics.accuracy_score(y_test2, y_test_pred))
    
    #CART
from sklearn.tree import DecisionTreeClassifier
for i in range (2,21):
    model_CART = DecisionTreeClassifier(max_depth=i)
    scores = cross_val_score(estimator=model_CART, X=X2, y=y2, cv=5) 
    print(i,':',np.average(scores))
    
    #GBT
from sklearn.ensemble import GradientBoostingClassifier 
for i in range (2,10):
    model_GBT = GradientBoostingClassifier(random_state=0,min_samples_split=i,n_estimators=100) 
    scores = cross_val_score(estimator=model_GBT, X=X2, y=y2, cv=5) 
    print(i,':',np.average(scores))
    
    #ANN
from sklearn.neural_network import MLPClassifier 
for i in range (2,11):
    model_ANN = MLPClassifier(hidden_layer_sizes=(i),max_iter=1000, random_state=0) 
    scores = cross_val_score(estimator=model_ANN, X=X2, y=y2, cv=5) 
    print(i,':',np.average(scores))
  
model_ANN2 = MLPClassifier(hidden_layer_sizes=(10,10),max_iter=1000, random_state=0) 
cross_val_score(estimator=model_ANN2, X=X2, y=y2, cv=5) 
model_ANN2.fit(X_train_std2,y_train2)
y_test_pred_2 = model_ANN2.predict(X_test2)
metrics.accuracy_score(y_test2, y_test_pred_2)



 ######################################### Overfitting Test
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.model_selection import cross_val_score 


df_grading_sample = pandas.read_excel("Kickstarter-Result.xlsx")
    
#data preprocessing
df_grading_sample = df_grading_sample.loc[ (df_grading_sample['state'] == 'failed') | (df_grading_sample['state'] == 'successful') ]

    #delete invalid variables
df_grading_sample = df_grading_sample.drop(columns = ['project_id','name','pledged'])
df_grading_sample = df_grading_sample.drop(columns = ['staff_pick','spotlight','backers_count'])
df_grading_sample = df_grading_sample.drop(columns = ['state_changed_at','state_changed_at_weekday','state_changed_at_month','state_changed_at_day','state_changed_at_yr','state_changed_at_hr','launch_to_state_change_days'])

    #delete duplicates
df_grading_sample.drop_duplicates()

    #manual feature selection & transformation 
df_grading_sample['disable_communication'] = pandas.get_dummies(df_grading_sample['disable_communication']) #col2

df_grading_sample['bad_category'] = np.where((df_grading_sample['category'] == 'Musical')|(df_grading_sample['category'] == 'Plays'), True, False)
df_grading_sample['bad_category'] = pandas.get_dummies(df_grading_sample['bad_category']) #col32

df_grading_sample['created_on_Sunday'] = np.where(df_grading_sample['created_at_weekday']== 'Sunday', True, False)
df_grading_sample['created_on_Sunday'] = pandas.get_dummies(df_grading_sample['created_on_Sunday']) #col33

df_grading_sample['launched_on_weekend'] = np.where((df_grading_sample['launched_at_weekday']== 'Sunday')|(df_grading_sample['launched_at_weekday']== 'Saturday')|(df_grading_sample['launched_at_weekday']== 'Friday'), True, False)
df_grading_sample['launched_on_weekend'] = pandas.get_dummies(df_grading_sample['launched_on_weekend']) #col34

df_grading_sample['deadline_yr_12-16'] = np.where((df_grading_sample['deadline_yr']<= 2016)&(df_grading_sample['deadline_yr']>= 2012), True, False)
df_grading_sample['deadline_yr_12-16'] = pandas.get_dummies(df_grading_sample['deadline_yr_12-16']) #col35

df_grading_sample['created_yr_12-16'] = np.where((df_grading_sample['created_at_yr']<= 2016)&(df_grading_sample['created_at_yr']>= 2012), True, False)
df_grading_sample['created_yr_12-16'] = pandas.get_dummies(df_grading_sample['created_yr_12-16']) #col36

df_grading_sample['launch_to_deadline_over60'] = np.where(df_grading_sample['launch_to_deadline_days']>60, True, False)
df_grading_sample['launch_to_deadline_over60'] = pandas.get_dummies(df_grading_sample['launch_to_deadline_over60']) #col37

df_grading_sample['create_to_launch_days_long'] = np.where(df_grading_sample['create_to_launch_days']>1000, True, False)
df_grading_sample['create_to_launch_days_long'] = pandas.get_dummies(df_grading_sample['create_to_launch_days_long']) #col38

df_grading_sample['create_to_launch_days_short'] = np.where(df_grading_sample['create_to_launch_days']<500, True, False)
df_grading_sample['create_to_launch_days_short'] = pandas.get_dummies(df_grading_sample['create_to_launch_days_short']) #col39

    #Regression
X_sample = df_grading_sample.iloc[:,[0,2,8,11,12,13,14,32,33,34,35,36,37,38,39]]
y_sample = df_grading_sample["usd_pledged"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_sample, y_sample, test_size = 0.33, random_state = 5)
scaler = StandardScaler()
X_train_std_s = scaler.fit_transform(X_train_s)

for i in range (500,5500,500):
    opt_model_linear_s = Lasso(alpha=i)
    opt_model_linear_s.fit(X_train_std_s,y_train_s) 
    y_test_pred = opt_model_linear_s.predict(X_test_s)
    print('Alpha = ',i,' / MSE =',mean_squared_error(y_test_s, y_test_pred))

opt_model_linear_s = Lasso(alpha=1500)
opt_model_linear_s.fit(X_train_std_s,y_train_s) 
opt_model_linear_s.coef_
pandas.DataFrame(list(zip(X_sample.columns,opt_model_linear_s.coef_)), columns = ['predictor','coefficient'])

    #Classification
X_sample2 = df_grading_sample.iloc[:,[0,2,8,11,12,13,14,32,33,34,35,36,37,38,39]]
y_sample2 = df_grading_sample["state"]

X_train_s2, X_test_s2, y_train_s2, y_test_s2 = train_test_split(X_sample2, y_sample2, test_size = 0.33, random_state = 5)
scaler = StandardScaler()
X_train_std_s2 = scaler.fit_transform(X_train_s2)

    #CART
model_CART = DecisionTreeClassifier(max_depth=4)
scores = cross_val_score(estimator=model_CART, X=X_sample2, y=y_sample2, cv=5) 
print(i,':',np.average(scores))

    #GBT
model_GBT = GradientBoostingClassifier(random_state=0,min_samples_split=7,n_estimators=100) 
scores = cross_val_score(estimator=model_GBT, X=X_sample2, y=y_sample2, cv=5) 
print(i,':',np.average(scores))
    
    #random forest
model_randomforest = RandomForestClassifier(random_state=5) 
scores = cross_val_score(estimator=model2, X=X_sample2, y=y_sample2, cv=5) 
print(i,':',np.average(scores))




 ######################################### C.Clustering
from sklearn.metrics import silhouette_score

df_starter = pandas.read_excel("P:/INSY446/project final/Kickstarter.xlsx")
df_starter = df_starter.loc[ (df_starter['state'] == 'failed') ]
X_cluster = df_starter.iloc[:,[19,20,21,28,32,36,40,42]]
X_cluster.info()
X_cluster = X_cluster.dropna()
X_cluster = X_cluster.astype(np.float64)
X_cluster.info()
X_std = scaler.fit_transform(X_cluster)

###K-Mean Clustering
from sklearn.cluster import KMeans
for i in range (2,8):
    kmeans = KMeans(n_clusters=i) 
    model = kmeans.fit(X_std)
    labels = model.predict(X_std)
    print(silhouette_score(X_std,labels) )

kmeans_k3 = KMeans(n_clusters=3) 
model_k3 = kmeans_k3.fit(X_std)
labels_k3 = model_k3.predict(X_std)
kmeans_k3.cluster_centers_
    
kmeans_k4 = KMeans(n_clusters=4) 
model_k4 = kmeans_k4.fit(X_std)
labels_k4 = model_k4.predict(X_std)
kmeans_k4.cluster_centers_

kmeans_k5 = KMeans(n_clusters=5) 
model_k5 = kmeans_k4.fit(X_std)
labels_k5 = model_k4.predict(X_std)
kmeans_k5.cluster_centers_
    ###k=3 shows the most reasonable & meaningful separation

 ######################################### Grading Part
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.model_selection import cross_val_score 

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingClassifier 



df_grading = pandas.read_excel("P:/INSY446/project final/Kickstarter-Grading-Sample.xlsx")
    
#data preprocessing
df_grading = df_grading.loc[ (df_grading['state'] == 'failed') | (df_grading['state'] == 'successful') ]

    #delete invalid variables
df_grading = df_grading.drop(columns = ['project_id','name','pledged'])
df_grading = df_grading.drop(columns = ['staff_pick','spotlight','backers_count'])
df_grading = df_grading.drop(columns = ['state_changed_at','state_changed_at_weekday','state_changed_at_month','state_changed_at_day','state_changed_at_yr','state_changed_at_hr','launch_to_state_change_days'])

    #delete duplicates
df_grading.drop_duplicates()

    #manual feature selection & transformation 
df_grading['disable_communication'] = pandas.get_dummies(df_grading['disable_communication']) #col2

df_grading['bad_category'] = np.where((df_grading['category'] == 'Musical')|(df_grading['category'] == 'Plays'), True, False)
df_grading['bad_category'] = pandas.get_dummies(df_grading['bad_category']) #col32

df_grading['created_on_Sunday'] = np.where(df_grading['created_at_weekday']== 'Sunday', True, False)
df_grading['created_on_Sunday'] = pandas.get_dummies(df_grading['created_on_Sunday']) #col33

df_grading['launched_on_weekend'] = np.where((df_grading['launched_at_weekday']== 'Sunday')|(df_grading['launched_at_weekday']== 'Saturday')|(df_grading['launched_at_weekday']== 'Friday'), True, False)
df_grading['launched_on_weekend'] = pandas.get_dummies(df_grading['launched_on_weekend']) #col34

df_grading['deadline_yr_12-16'] = np.where((df_grading['deadline_yr']<= 2016)&(df_grading['deadline_yr']>= 2012), True, False)
df_grading['deadline_yr_12-16'] = pandas.get_dummies(df_grading['deadline_yr_12-16']) #col35

df_grading['created_yr_12-16'] = np.where((df_grading['created_at_yr']<= 2016)&(df_grading['created_at_yr']>= 2012), True, False)
df_grading['created_yr_12-16'] = pandas.get_dummies(df_grading['created_yr_12-16']) #col36

df_grading['launch_to_deadline_over60'] = np.where(df_grading['launch_to_deadline_days']>60, True, False)
df_grading['launch_to_deadline_over60'] = pandas.get_dummies(df_grading['launch_to_deadline_over60']) #col37

df_grading['create_to_launch_days_long'] = np.where(df_grading['create_to_launch_days']>1000, True, False)
df_grading['create_to_launch_days_long'] = pandas.get_dummies(df_grading['create_to_launch_days_long']) #col38

df_grading['create_to_launch_days_short'] = np.where(df_grading['create_to_launch_days']<500, True, False)
df_grading['create_to_launch_days_short'] = pandas.get_dummies(df_grading['create_to_launch_days_short']) #col39

#separate data - reg
X = df_grading.iloc[:,[0,2,8,11,12,13,14,32,33,34,35,36,37,38,39]]
y = df_grading["usd_pledged"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 5)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)

opt_model_linear = Lasso(alpha=1500)
opt_model_linear.fit(X_train_std,y_train) 
#opt_model_linear.coef_
#pandas.DataFrame(list(zip(X.columns,opt_model_linear.coef_)), columns = ['predictor','coefficient'])

y_test_pred = opt_model_linear.predict(X_test)
print(mean_squared_error(y_test, y_test_pred))

#separate data - classification
y = df_grading["state"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 5)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)

model_GBT = GradientBoostingClassifier(random_state=0,min_samples_split=7,n_estimators=100) 
model_GBT.fit(X_train, y_train)
y_test_pred = model_GBT.predict(X_test)
metrics.accuracy_score(y_test, y_test_pred)

scores = cross_val_score(estimator=model_GBT, X=X, y=y, cv=5) 
print(np.average(scores))
    
  










