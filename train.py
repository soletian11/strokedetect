import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,auc
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,roc_curve
from sklearn.metrics import mean_squared_error 

#Global Vars
max_depth=5
min_samples_leaf=21 
estimators=210
C=10
model_file=f'model_rf.bin'


#Reading Data

data='https://raw.githubusercontent.com/soletian11/strokedetect/main/healthcare-dataset-stroke-data.csv'
df=pd.read_csv(data)
# ## Standardization 
#     1.Having all lower case and underscores for col name having ''
#     2.Having data in all lower case and underscores having ''

# In[7]:


#Cleaning metadata like column names having every column lower case 
#Categorcial Columns and Numerical Cols
numerical=[ 'age',  'hypertension',  'heart_disease',  'avg_glucose_level',  'bmi']
categorical=['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status']

df=pd.read_csv(data)
df.columns=df.columns.str.lower()

## Cleaning up data and makes values lower case in of categorical

df=df[df['gender']!='Other'].reset_index(drop=True)
df['age']=df['age'].round()

for col in categorical:
    df[col]=df[col].str.lower().str.replace(' ','_')

# ##Adding rules on numerical col's
# ##Age :rounding  because it has decimals values like 0.8 and added new col age_modified
# df['age_modified']=df['age'].round()
# df[df['age']<10]
# ####

# ##bmi :setting 0 for NAN
# df['bmi']=df['bmi'].fillna(0)
df['bmi']=df['bmi'].fillna(df['bmi'].mean())

#Training
df_full_train, df_test=train_test_split(df,test_size=0.2,random_state=42)
df_train, df_val=train_test_split(df_full_train,test_size=0.25,random_state=42)
df_train=df_train.reset_index(drop=True)
df_val=df_val.reset_index(drop=True)
df_test=df_test.reset_index(drop=True)
y_train=df_train['stroke'].values
y_full_train=df_full_train['stroke'].values
y_test=df_test['stroke'].values
y_val=df_val['stroke'].values

del df_train['stroke']
del df_test['stroke']
del df_val['stroke']
 ## Feature Importance -Categorical


# train_dicts=df_train[categorical+numerical].to_dict(orient='records')
# test_dicts=df_test[categorical+numerical].to_dict(orient='records')
# val_dicts=df_val[categorical+numerical].to_dict(orient='records')
# dv=DictVectorizer(sparse=False)
# X_train=dv.fit_transform(train_dicts)
# X_val=dv.transform(val_dicts)
# X_test=dv.transform(test_dicts)


def train(df_train,y_train,C):
    train_dicts=df_train[numerical+categorical].to_dict(orient='records')
    dv=DictVectorizer(sparse=False)
    X_train=dv.fit_transform(train_dicts)
    model_rf=RandomForestClassifier(n_estimators=estimators,
                                    random_state=1,
                                    max_depth=max_depth,
                                    n_jobs=2,
                                    min_samples_leaf=min_samples_leaf,
                                    max_features=max_features
              )
    model_rf.fit(X_train,y_train)
    
    
    return model_rf,dv

def predict(df,dv,model):
    dicts=df[numerical+categorical].to_dict(orient='records')
    X=dv.transform(dicts)
    y_pred=model.predict_proba(X)[:,1]
    
    return y_pred
    


print("Start Training of Model")
model,dv=train(df_train,y_train, C=C)
y_pred=predict(df_val,dv,model)



with open(model_file,'wb') as f_out:
    pickle.dump((dv,model),f_out)

with open (model_file,'rb') as f_in:
    dv,model=pickle.load(f_in)



patient={"gender": "female",
  "ever_married": "yes",
  "work_type": "self-employed",
  "residence_type": "rural",
  "smoking_status": "smoked",
  "id": 57043,
  "age": 88.0,
  "hypertension": 0,
  "heart_disease": 0,
  "avg_glucose_level": 102.73,
  "bmi": 35.0}

print("Transform and testing")
X=dv.transform(patient)

stroke_predict=model.predict_proba(X)[:,1]
print(stroke_predict)

