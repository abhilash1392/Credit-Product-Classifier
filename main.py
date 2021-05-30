# Importing the libraries
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import make_column_transformer 
from sklearn.pipeline import make_pipeline 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

# Importing the file
df_train = pd.read_csv('train_s3TEQDk.csv')
df_train=df_train.sample(frac=1).reset_index(drop=True)
print('Train file loaded.')
df_train['Credit_Product'] = df_train['Credit_Product'].replace(np.nan,'NoInformation')
features = [f for f in df_train.columns if f not in ('ID','Region_Code','Is_Lead')]
df_train,df_valid = train_test_split(df_train,test_size=0.2,stratify=df_train.Is_Lead.values)
xtrain = df_train[features]
ytrain = df_train.Is_Lead.values
xvalid = df_valid[features]
yvalid = df_valid.Is_Lead.values
categorial_columns = [f for f in features if df_train[f].dtypes=='O']
numerical_columns = [f for f in features if df_train[f].dtypes!='O']
print('Data Transformation Done')
transformer = make_column_transformer((StandardScaler(),numerical_columns),(OneHotEncoder(),categorial_columns))
param_grid = {
    'randomforestclassifier__bootstrap': [True],
    'randomforestclassifier__max_depth': [80, 90, 100, 110],
    'randomforestclassifier__max_features': [2, 3],
    'randomforestclassifier__min_samples_leaf': [3, 4, 5],
    'randomforestclassifier__min_samples_split': [8, 10, 12],
    'randomforestclassifier__n_estimators': [100, 200, 300, 1000]}

model = RandomForestClassifier()

clf = make_pipeline(transformer,model)
# print(clf.get_params().keys())
grid_search =  GridSearchCV(estimator=clf,param_grid=param_grid,cv=5,n_jobs=-1,verbose=2)
grid_search.fit(xtrain,ytrain)
print('Model fitted')
print('Best Accuracy: {}'.format(grid_search.best_score_))
print('Best Params: {}'.format(grid_search.best_params_))
print('Validation Score: {}'.format(grid_search.score(xvalid,yvalid)))
df_test = pd.read_csv('test_mSzZ8RL.csv')
df_test['Credit_Product'] = df_test['Credit_Product'].replace(np.nan,'NoInformation')
xtest = df_test[features]
test_id=df_test['ID']
ytest = grid_search.predict(xtest)
print('Prediction done')
data = {'ID':test_id,'Is_Lead':ytest}
df_submission = pd.DataFrame(data)
df_submission.to_csv('submission.csv',index=False)