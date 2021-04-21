#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df =  pd.read_csv(r'C:\Users\User\Downloads\archive (1)\Churn_Modelling.csv')
df.sample(5)


# In[3]:


#retirar colunas
df.drop(['RowNumber', 'CustomerId', 'Surname'],axis='columns',inplace=True)


# In[4]:


df.shape


# In[5]:


#checagem de dados faltantes
faltantes = df.isnull().sum()
faltantes


# In[6]:


#checar colunas que são fatores
def print_unique_col_values(df):
       for column in df:
            if df[column].dtypes=='object':
                print(f'{column}: {df[column].unique()}')


# In[7]:


print_unique_col_values(df)


# In[8]:


#dummy na variável sex
df.replace('Male',1,inplace=True)
df.replace('Female',0,inplace=True)


# In[9]:


print_unique_col_values(df)


# In[10]:


#trasformar em dummy as variáveis categóricas
df_ready = pd.get_dummies(data=df, columns=['Geography'], drop_first=True) #drop_first to avoid the dummy trap
df_ready.columns


# In[11]:


print_unique_col_values(df_ready)


# In[12]:


df_ready.head(5)


# In[13]:


#escalonar variáveis
cols_to_scale = ['Age','CreditScore','Balance','Tenure', 'EstimatedSalary']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_ready[cols_to_scale] = scaler.fit_transform(df_ready[cols_to_scale])


# In[14]:


#checar desbalanceamento
df_ready['Exited'].value_counts()


# In[15]:


#separando em treino e teste
X = df_ready.drop('Exited',axis='columns')
y = df_ready['Exited']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)


# In[ ]:


#treinando o modelo


# In[16]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rnd_clf = RandomForestClassifier(random_state=42)
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)
accuracy_score(y_test, y_pred_rf)


# In[17]:


#acuracia, recall, precisao e f1-score
from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test,y_pred_rf))


# In[18]:


#matriz de confusao
import tensorflow as tf
import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred_rf)

plt.figure(figsize = (5,4))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')


# In[19]:


#balanceando os dados


# In[20]:


from imblearn.over_sampling import SMOTE


# In[21]:


y.value_counts() #antes de balancear


# In[22]:


from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority')
X_smote, y_smote = smote.fit_sample(X,y)

y_smote.value_counts() #após balancear


# In[34]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_smote,y_smote,test_size=0.2, stratify = y_smote)


# In[24]:


y_test.value_counts()


# In[35]:


#ajustando novamente o modelo
rnd_clf = RandomForestClassifier(random_state=42)
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)
accuracy_score(y_test, y_pred_rf)


# In[36]:


from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test,y_pred_rf))


# In[37]:


import tensorflow as tf
import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred_rf)

plt.figure(figsize = (5,4))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')


# In[38]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 1, stop = 200, num = 20)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 50, num =10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [1,2,5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1,2,5]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#usar o randomizedsearchcv para o tunning dos hiperparametros


# In[ ]:





# In[43]:


rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter =1000 , cv = 3, verbose=True, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train,y_train)


# In[44]:


rf_random.best_params_


# In[39]:


rf2 = RandomForestClassifier(n_estimators= 179,
 min_samples_split= 2,
 min_samples_leaf=1,
 max_features= 'auto',
 max_depth= 33,
 bootstrap= False,
       random_state=42                      
)


# In[40]:


rf2.fit(X_train, y_train)
y_predicao = rf2.predict(X_test)
print(accuracy_score(y_test, y_predicao))


# In[56]:



print(classification_report(y_test,y_predicao))


# In[ ]:




