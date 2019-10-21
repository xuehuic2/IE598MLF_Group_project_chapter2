#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv('MLF_GP2_EconCycle.csv')
df.head()


# In[3]:


#df2=df2.dropna()
df = df.iloc[:,1:]#drop the date column 
df['CP1M_T2Y']= df['CP1M']/df['T2Y Index']#add more column about commercial paper and other Treasury bill spread 
df['CP3M_T2Y']= df['CP3M']/df['T2Y Index']
df['CP6M_T2Y']= df['CP6M']/df['T2Y Index']

df['CP1M_T3Y']= df['CP1M']/df['T3Y Index']
df['CP3M_T3Y']= df['CP3M']/df['T3Y Index']
df['CP6M_T3Y']= df['CP6M']/df['T3Y Index']

df['CP1M_T5Y']= df['CP1M']/df['T5Y Index']
df['CP3M_T5Y']= df['CP3M']/df['T5Y Index']
df['CP6M_T5Y']= df['CP6M']/df['T5Y Index']

df['CP1M_T7Y']= df['CP1M']/df['T7Y Index']
df['CP3M_T7Y']= df['CP3M']/df['T7Y Index']
df['CP6M_T7Y']= df['CP6M']/df['T7Y Index']

df['CP1M_T10Y']= df['CP1M']/df['T10Y Index']
df['CP3M_T10Y']= df['CP3M']/df['T10Y Index']
df['CP6M_T10Y']= df['CP6M']/df['T10Y Index']


# In[4]:


df.head()


# In[21]:


#cols=['T1Y Index','CP3M','CP3M_T1Y','PCT 3MO FWD']
#columns' name for features
Xcols=['T1Y Index','T2Y Index','T3Y Index','T5Y Index','T7Y Index','T10Y Index','CP1M','CP3M','CP6M','CP1M_T1Y','CP3M_T1Y','CP6M_T1Y',
       'CP1M_T2Y','CP3M_T2Y','CP6M_T2Y',
      'CP1M_T3Y','CP3M_T3Y','CP6M_T3Y',
      'CP1M_T5Y','CP3M_T5Y','CP6M_T5Y',
      'CP1M_T7Y','CP3M_T7Y','CP6M_T7Y',
      'CP1M_T10Y','CP3M_T10Y','CP6M_T10Y']
ycols = ['PCT 3MO FWD','PCT 6MO FWD','PCT 9MO FWD']
#HeatMap
cols=['CP1M_T1Y','CP3M_T1Y','CP6M_T1Y',
      #'CP1M_T2Y',
      #'CP1M_T3Y',
      #'CP1M_T5Y',
      #'CP1M_T7Y',
      #'CP1M_T10Y',
      'PCT 3MO FWD','PCT 6MO FWD','PCT 9MO FWD']
#Scatterplot Matrix
sns.pairplot(df[cols], height=2)
plt.show()


# In[17]:


import seaborn as sns
cm = np.corrcoef(df[cols].values.T)
ax = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols, xticklabels=cols) #notation: "annot" not "annote"
bottom, top = ax.get_ylim()
plt.title("Heatmap Commercial/Treasury ratio and percentage change USPHCI")
ax.set_ylim(bottom + 0.5, top - 0.5)


# In[88]:


#Preprocessing, Feature extraction and Feature selection
X=df[Xcols]
y=df[ycols]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=42)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#X_train_std = StandardScaler().fit_transform(X_train)
#X_test_std= StandardScaler().fit_transform(X_test)

X_train_std=StandardScaler().fit_transform(X_train)
X_test_std=StandardScaler().fit_transform(X_test)

y_train_std=StandardScaler().fit_transform(y_train)
#y_train_std=y_train
y_test_std = StandardScaler().fit_transform(y_test)
#y_test_std = y_test


# In[89]:


#Using ridge regression to modeling
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#k_range = np.arange(0.0, 1.0, 0.1)
k_range = (0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)
for k in k_range:
    ridge = Ridge(alpha=k)
    ridge.fit(X_train, y_train)
    y_train_pred = ridge.predict(X_train_std)
    y_test_pred = ridge.predict(X_test_std)
    print('alpha=',k,'MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train_std, y_train_pred),
        mean_squared_error(y_test_std, y_test_pred)))
    print('alpha=',k ,'R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train_std, y_train_pred),
        r2_score(y_test_std, y_test_pred)))


# In[90]:


#coef_
ridge = Ridge(alpha=1)
ridge.fit(X_train_std, y_train_std)
y_train_pred = ridge.predict(X_train_std)
y_test_pred = ridge.predict(X_test_std)
#print(ridge.coef_)
#ridge.coef_
df_4 = pd.DataFrame(ridge.coef_)
df_2 = pd.DataFrame(Xcols)
coefficient3= pd.concat([df_2, df_4.iloc[0],df_4.iloc[1],df_4.iloc[2]],axis=1, ignore_index=True)
coefficient3


# In[91]:


#LASSO regression
from sklearn.linear_model import Lasso
#k_range = np.arange(0.0, 1.0, 0.1)
k_range = (0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)
for k in k_range:
    lasso = Lasso(alpha=k)
    lasso.fit(X_train_std, y_train_std)
    y_train_pred = lasso.predict(X_train_std)
    y_test_pred = lasso.predict(X_test_std)
    print('alpha=',k,'MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train_std, y_train_pred),
        mean_squared_error(y_test_std, y_test_pred)))
    print('alpha=',k ,'R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train_std, y_train_pred),
        r2_score(y_test_std, y_test_pred)))

#here we get while alpha=0.0, we have a small MSE and large R^2


# In[92]:


lasso = Lasso(alpha=0.0)
lasso.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)
#print(lasso.coef_)
df_3 = pd.DataFrame(lasso.coef_)
df_2 = pd.DataFrame(Xcols)
coefficient3= pd.concat([df_2, df_3.iloc[0],df_3.iloc[1],df_3.iloc[2]],axis=1, ignore_index=True)
coefficient3
#here we can choose three components to predict our future 


# In[93]:


##using PCA to select features
#PCA
pca=PCA()
X_train_pca=pca.fit_transform(X_train_std)
#X_test_pca=pca.transform(X_test_std)

features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()
print(np.cumsum(pca.explained_variance_ratio_))


# In[95]:


#We choose the PCA components as 4 so that we can have 99% of the variance
#PCA
pca=PCA(n_components = 4)
X_train_pca=pca.fit_transform(X_train_std)
X_test_pca=pca.transform(X_test_std)
#print(X_train_pca)
#print(X_test_pca)
#y1_train= y_train_std.iloc[:,0]
#y2_train=y_train_std.iloc[:,1]
#y3_train=y_train_std.iloc[:,2]
y1_train= y_train_std[:,0]
y2_train=y_train_std[:,1]
y3_train=y_train_std[:,2]
#y1_test=y_test_std.iloc[:,0]
#y2_test=y_test_std.iloc[:,1]
#y3_test=y_test_std.iloc[:,2]
y1_test=y_test_std[:,0]
y2_test=y_test_std[:,1]
y3_test=y_test_std[:,2]
print(np.cumsum(pca.explained_variance_ratio_))


# In[76]:


#linear regression
from sklearn import linear_model
from sklearn.metrics import *
from math import *
#for y1 target

lr= linear_model.SGDRegressor(loss='squared_loss', penalty=None,random_state = 42)
lr.fit(X_train_pca, y1_train)

y1_pred_train=lr.predict(X_train_pca)
y1_pred_test=lr.predict(X_test_pca)

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y1_train, y1_pred_train),
        mean_squared_error(y1_test, y1_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y1_train, y1_pred_train),
        r2_score(y1_test, y1_pred_test)))
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y1_train, y1_pred_train)),
       sqrt(mean_squared_error(y1_test, y1_pred_test))))
print('Slope: ', lr.coef_)
print('Intercept: %.3f' % lr.intercept_)
#we don't get any linear correlation between the features and y1 targets
#which is evident from the heatmap also. 
#we do not try with normal linear, lasso and ridge regression because they are also linear regressors and the error 
#will remain the same as in the linear one


# In[77]:


#for y2 target
lr= linear_model.SGDRegressor(loss='squared_loss', penalty=None,random_state = 42)
lr.fit(X_train_pca, y2_train)
y2_pred_train=lr.predict(X_train_pca)
y2_pred_test=lr.predict(X_test_pca)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y2_train, y2_pred_train),
        mean_squared_error(y2_test, y2_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y2_train, y2_pred_train),
        r2_score(y2_test, y2_pred_test)))
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y2_train, y2_pred_train)),
       sqrt(mean_squared_error(y2_test, y2_pred_test))))
print('Slope: ', lr.coef_)
print('Intercept: %.3f' % lr.intercept_)


# In[78]:


#for y3 target
lr= linear_model.SGDRegressor(loss='squared_loss', penalty=None,random_state = 42)
lr.fit(X_train_pca, y3_train)
y3_pred_train=lr.predict(X_train_pca)
y3_pred_test=lr.predict(X_test_pca)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y3_train, y3_pred_train),
        mean_squared_error(y3_test, y3_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y3_train, y3_pred_train),
        r2_score(y3_test, y3_pred_test)))
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y3_train, y3_pred_train)),
       sqrt(mean_squared_error(y3_test, y3_pred_test))))
print('Slope: ', lr.coef_)
print('Intercept: %.3f' % lr.intercept_)


# In[79]:


#DecisionTreeRegressor FOR Y1
from sklearn.tree import DecisionTreeRegressor

# Instantiate dt
dt = DecisionTreeRegressor(max_depth=8,
             min_samples_leaf=0.13,
            random_state=3)

# Fit dt to the training set
dt.fit(X_train_pca, y1_train)

# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE
y1_pred_train = dt.predict(X_train_pca)
y1_pred_test = dt.predict(X_test_pca)

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y1_train, y1_pred_train),
        mean_squared_error(y1_test, y1_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y1_train, y1_pred_train),
        r2_score(y1_test, y1_pred_test)))
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y1_train, y1_pred_train)),
       sqrt(mean_squared_error(y1_test, y1_pred_test))))


# In[80]:


#DecisionTreeRegressor FOR Y2
from sklearn.tree import DecisionTreeRegressor
# Instantiate dt
dt = DecisionTreeRegressor(max_depth=8,
             min_samples_leaf=0.13,
            random_state=3)

# Fit dt to the training set
dt.fit(X_train_pca, y2_train)

# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE
y2_pred_train = dt.predict(X_train_pca)
y2_pred_test = dt.predict(X_test_pca)

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y2_train, y2_pred_train),
        mean_squared_error(y2_test, y2_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y2_train, y2_pred_train),
        r2_score(y2_test, y2_pred_test)))
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y2_train, y2_pred_train)),
       sqrt(mean_squared_error(y2_test, y2_pred_test))))


# In[81]:


#DecisionTreeRegressor FOR Y3
from sklearn.tree import DecisionTreeRegressor
# Instantiate dt
dt = DecisionTreeRegressor(max_depth=8,
             min_samples_leaf=0.13,
            random_state=3)

# Fit dt to the training set
dt.fit(X_train_pca, y3_train)

# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE
y3_pred_train = dt.predict(X_train_pca)
y3_pred_test = dt.predict(X_test_pca)

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y3_train, y3_pred_train),
        mean_squared_error(y3_test, y3_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y3_train, y3_pred_train),
        r2_score(y3_test, y3_pred_test)))
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y3_train, y3_pred_train)),
       sqrt(mean_squared_error(y3_test, y3_pred_test))))




# In[98]:


#using support vector regressor to fit model
from sklearn.svm import SVR
#for y1
SVR = SVR(kernel="linear").fit(X_train_pca, y1_train)
SVR.predict(X_test_pca)

y1_train_pred = SVR.predict(X_train_pca)
y1_test_pred = SVR.predict(X_test_pca)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y1_train, y1_train_pred),
        mean_squared_error(y1_test, y1_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y1_train, y1_train_pred),
        r2_score(y1_test, y1_test_pred)))
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y1_train, y1_train_pred)),
       sqrt(mean_squared_error(y1_test, y1_test_pred))))


# In[99]:


from sklearn.svm import SVR
#for y2
SVR = SVR(kernel="linear").fit(X_train_pca, y2_train)
SVR.predict(X_test_pca)

y2_train_pred = SVR.predict(X_train_pca)
y2_test_pred = SVR.predict(X_test_pca)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y2_train, y2_train_pred),
        mean_squared_error(y2_test, y2_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y2_train, y2_train_pred),
        r2_score(y2_test, y2_test_pred)))
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y2_train, y2_train_pred)),
       sqrt(mean_squared_error(y2_test, y2_test_pred))))


# In[100]:


from sklearn.svm import SVR
#for y3
SVR = SVR(kernel="linear").fit(X_train_pca, y3_train)
SVR.predict(X_test_pca)

y3_train_pred = SVR.predict(X_train_pca)
y3_test_pred = SVR.predict(X_test_pca)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y3_train, y3_train_pred),
        mean_squared_error(y3_test, y3_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y3_train, y3_train_pred),
        r2_score(y3_test, y3_test_pred)))
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y3_train, y3_train_pred)),
       sqrt(mean_squared_error(y3_test, y3_test_pred))))


# In[101]:


#using random forest regressor to fit the model 
#for y1 target
from sklearn.ensemble import RandomForestRegressor
# Instantiate rf
rf = RandomForestRegressor(n_estimators=200,
            random_state=42)
            
# Fit rf to the training set    
rf.fit(X_train_pca, y1_train) 
y1_pred_train=rf.predict(X_train_pca)
y1_pred_test=rf.predict(X_test_pca)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y1_train, y1_pred_train),
        mean_squared_error(y1_test, y1_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y1_train, y1_pred_train),
        r2_score(y1_test, y1_pred_test)))
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y1_train, y1_pred_train)),
       sqrt(mean_squared_error(y1_test, y1_pred_test))))



# In[102]:


#using random forest regressor to fit the model 
#for y2 target
from sklearn.ensemble import RandomForestRegressor
# Instantiate rf
rf = RandomForestRegressor(n_estimators=200,
            random_state=42)
            
# Fit rf to the training set    
rf.fit(X_train_pca, y2_train) 
y2_pred_train=rf.predict(X_train_pca)
y2_pred_test=rf.predict(X_test_pca)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y2_train, y2_pred_train),
        mean_squared_error(y2_test, y2_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y2_train, y2_pred_train),
        r2_score(y2_test, y2_pred_test)))
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y2_train, y2_pred_train)),
       sqrt(mean_squared_error(y2_test, y2_pred_test))))



# In[103]:


#using random forest regressor to fit the model 
#for y3 target
from sklearn.ensemble import RandomForestRegressor
# Instantiate rf
rf = RandomForestRegressor(n_estimators=60,max_depth=7,
            random_state=42)
            
# Fit rf to the training set    
rf.fit(X_train_pca, y3_train) 
y3_pred_train=rf.predict(X_train_pca)
y3_pred_test=rf.predict(X_test_pca)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y3_train, y3_pred_train),
        mean_squared_error(y3_test, y3_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y3_train, y3_pred_train),
        r2_score(y3_test, y3_pred_test)))
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y3_train, y3_pred_train)),
       sqrt(mean_squared_error(y3_test, y3_pred_test))))


# In[104]:


params_rf = {'n_estimators':[50,100,350,500],'max_features':['log2', 'auto', 'sqrt'], 'min_samples_leaf':[2,10 , 30] }
from sklearn.model_selection import GridSearchCV
#for y1
# Instantiate grid_rf
grid_rf = GridSearchCV(estimator=rf,
                       param_grid=params_rf,
                       scoring='neg_mean_squared_error',
                       cv=10,
                       verbose=1,
                       n_jobs=-1)
grid_rf.fit(X_train_pca,y1_train)
best_model = grid_rf.best_estimator_
print(best_model)
# Predict test set labels
y1_train_pred = best_model.predict(X_train_pca)
y1_pred = best_model.predict(X_test_pca)
# Compute rmse_test
rmse_train = mean_squared_error(y1_train,y1_train_pred)**(1/2)
rmse_test = mean_squared_error(y1_test,y1_pred)**(1/2)

# Print rmse_test

print('Train RMSE of best model: {:.3f}'.format(rmse_test)) 
print('Test RMSE of best model: {:.3f}'.format(rmse_test))
print('R^2 train:', (r2_score(y1_train, y1_train_pred)))
print('R^2 test:', (r2_score(y1_test, y1_pred)))


# In[105]:


params_rf = {'n_estimators':[50,100,350,500],'max_features':['log2', 'auto', 'sqrt'], 'min_samples_leaf':[2,10 , 30] }
from sklearn.model_selection import GridSearchCV
#for y2
# Instantiate grid_rf
grid_rf = GridSearchCV(estimator=rf,
                       param_grid=params_rf,
                       scoring='neg_mean_squared_error',
                       cv=10,
                       verbose=1,
                       n_jobs=-1)
grid_rf.fit(X_train_pca,y2_train)
best_model = grid_rf.best_estimator_
print(best_model)
# Predict test set labels
y2_train_pred = best_model.predict(X_train_pca)
y2_pred = best_model.predict(X_test_pca)
# Compute rmse_test
rmse_train = mean_squared_error(y2_train,y2_train_pred)**(1/2)
rmse_test = mean_squared_error(y2_test,y2_pred)**(1/2)

# Print rmse_test

print('Train RMSE of best model: {:.3f}'.format(rmse_test)) 
print('Test RMSE of best model: {:.3f}'.format(rmse_test))
print('R^2 train:', (r2_score(y2_train, y2_train_pred)))
print('R^2 test:', (r2_score(y2_test, y2_pred)))



# In[439]:


params_rf = {'n_estimators':[50,60,100,350,500],'max_features':['log2', 'auto', 'sqrt'], 'min_samples_leaf':[2,10 , 30] }
from sklearn.model_selection import GridSearchCV
#for y3
# Instantiate grid_rf
grid_rf = GridSearchCV(estimator=rf,
                       param_grid=params_rf,
                       scoring='neg_mean_squared_error',
                       cv=10,
                       verbose=1,
                       n_jobs=-1)
grid_rf.fit(X_train_pca,y3_train)
best_model = grid_rf.best_estimator_
print(best_model)
# Predict test set labels
y3_train_pred = best_model.predict(X_train_pca)
y3_pred = best_model.predict(X_test_pca)
# Compute rmse_test
rmse_train = mean_squared_error(y3_train,y3_train_pred)**(1/2)
rmse_test = mean_squared_error(y3_test,y3_pred)**(1/2)

# Print rmse_test

print('Train RMSE of best model: {:.3f}'.format(rmse_test)) 
print('Test RMSE of best model: {:.3f}'.format(rmse_test))
print('R^2 train:', (r2_score(y3_train, y3_train_pred)))
print('R^2 test:', (r2_score(y3_test, y3_pred)))


# In[122]:


# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
#for y1
# Instantiate sgbr
sgbr = GradientBoostingRegressor(max_depth=4, 
            subsample=1,
            max_features=0.75,
            n_estimators=16,                                
            random_state=2)
# Fit sgbr to the training set
sgbr.fit(X_train_pca,y1_train)

# Predict test set labels

y1_train_pred = sgbr.predict(X_train_pca)
y1_test_pred = sgbr.predict(X_test_pca)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y1_train, y1_train_pred),
        mean_squared_error(y1_test, y1_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y1_train, y1_train_pred),
        r2_score(y1_test, y1_test_pred)))
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y1_train, y1_train_pred)),
       sqrt(mean_squared_error(y1_test, y1_test_pred))))


# In[107]:


# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Instantiate sgbr
sgbr = GradientBoostingRegressor(max_depth=4, 
            subsample=1,
            max_features=0.75,
            n_estimators=200,                                
            random_state=2)
# Fit sgbr to the training set
sgbr.fit(X_train_pca,y2_train)

# Predict test set labels

y2_train_pred = sgbr.predict(X_train_pca)
y2_test_pred = sgbr.predict(X_test_pca)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y2_train, y2_train_pred),
        mean_squared_error(y2_test, y2_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y2_train, y2_train_pred),
        r2_score(y2_test, y2_test_pred)))
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y2_train, y2_train_pred)),
       sqrt(mean_squared_error(y2_test, y2_test_pred))))


# In[108]:


# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Instantiate sgbr
sgbr = GradientBoostingRegressor(max_depth=7, min_samples_split = 5,
            min_samples_leaf = 1,          
            subsample=1,
            max_features=0.75,
            n_estimators=60,
            max_leaf_nodes = 14,
            random_state=42)
# Fit sgbr to the training set
sgbr.fit(X_train_pca,y3_train)

# Predict test set labels

y3_train_pred = sgbr.predict(X_train_pca)
y3_test_pred = sgbr.predict(X_test_pca)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y3_train, y3_train_pred),
        mean_squared_error(y3_test, y3_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y3_train, y3_train_pred),
        r2_score(y3_test, y3_test_pred)))
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y3_train, y3_train_pred)),
       sqrt(mean_squared_error(y3_test, y3_test_pred))))


# In[ ]:


#Machine learning group project
#Xuehui Chao
#Carolina Carvalho Manhães Leite
#Khavya Chandrasekaran

