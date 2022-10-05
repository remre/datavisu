import numpy as np
import pandas as pd 
import matplotlib as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error , mean_squared_log_error
from sklearn import set_config 
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,make_scorer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from joblib import dump,load

df = pd.read_csv('dbs/editedzomato.csv')

def one_hot_encode(df, column):
    # Get one hot encoding of columns B
    df[column] = pd.get_dummies(df[column],drop_first=True)
def one_hot_encoder_notboolean(df, column):
    # Get one hot encoding of columns B
    one_hot = pd.get_dummies(df[column])
    # Drop column as it is now encoded
    df = df.drop(column,axis = 1)
    # print(f"one hot encoded {column}")
    # Join the encoded df
    df = df.join(one_hot)
    return df
one_hot_colums = ['book_table','online_order']
for col in one_hot_colums:
    one_hot_encode(df,col)

df = one_hot_encoder_notboolean(df,'city')
df.drop(columns=['dish_liked','reviews_list','menu_item','type'], inplace  =True)
df['rest_type'] = df['rest_type'].str.replace(',' , '') 
df['rest_type'] = df['rest_type'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))
# df['rest_type'].value_counts().head()
df['cuisines'] = df['cuisines'].str.replace(',' , '') 
df['cuisines'] = df['cuisines'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))
# df['cuisines'].value_counts().head()
x = df.drop(['rate','name'],axis = 1)
y = df['rate']

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)


X_train =X_train.drop(['location','rest_type','cuisines',],axis = 1)
X_test =X_test.drop(['location','rest_type','cuisines',],axis = 1)


def mse(y, y_pred):
    return np.mean((y_pred - y)**2) 
def test_predict(model,X_train,X_test,y_train,y_test, parameters = None):
    model.fit(X_train, y_train)
    prediction_test = model.predict(X_test)
    model_text_list=[]; metric_list=[]; score_list=[] ; param_list=[]
    
    # create list of metric to be examined
    metric_functions = [r2_score, r2_score, mean_squared_error,mean_squared_error,mean_absolute_error]
    metric_functions_text = ['R_Squared', 'Adj_R_Squared', 'MSE','RMSE','MAE']
    
    # for loop of each of the 5 metrics
    for metric_function, metric_function_text in zip(metric_functions, metric_functions_text):
        if metric_function_text == 'Adj_R_Squared':
            Adj_r2 = 1 - (1-r2_score(y_test, prediction_test)) * (len(y)-1)/(len(y)-X_test.shape[1]-1)
            model_text_list.append(type(model).__name__); metric_list.append(metric_function_text); score_list.append(Adj_r2); param_list.append(parameters)
        elif metric_function_text == 'RMSE':
            rmse = mean_squared_error(y_test, prediction_test, squared=False)
            model_text_list.append(type(model).__name__); metric_list.append(metric_function_text); score_list.append(rmse); param_list.append(parameters)
        else:
            model_text_list.append(type(model).__name__); metric_list.append(metric_function_text); score_list.append(metric_function(y_test, prediction_test)); param_list.append(parameters) 
    
    d = {'model':model_text_list, 'parameters': param_list ,'metric': metric_list, 'test predict score': score_list}
    df = pd.DataFrame(data=d)
    return df
def five_cv_prarm_grid(PARAM_DICT, ESTIMATOR,X_train,y_train):
    sh = HalvingGridSearchCV(ESTIMATOR, PARAM_DICT, cv=10, scoring='neg_mean_absolute_error',min_resources="smallest",random_state=42).fit(X_train, y_train)
    best_estimator = sh.best_estimator_
    best_param = sh.best_params_
    print(best_estimator)
    print(f"10-CV Best Parameters = {best_param}")
    print(f"10-CV Best Score = {sh.best_score_}")
    return best_estimator, best_param



model_dtr = DecisionTreeRegressor(ccp_alpha=0.0, criterion='mse', max_depth=None,
                      max_features=None, max_leaf_nodes=None,
                      min_impurity_decrease=0.0, 
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0,
                      random_state=None, splitter='best') 
DTR_test = test_predict(model_dtr, X_train,X_test,y_train,y_test)
DTR_test
# mse_scorer = make_scorer(mse, greater_is_better=False)
# lr = LinearRegression()
# lr.fit(X_train,y_train)
# y_pred_lr = lr.predict(X_test)

# print(mse(y_test, y_pred_lr))
# print(X_train[:1])
# print(X_train.info())