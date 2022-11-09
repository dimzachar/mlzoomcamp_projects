# MLZoomcamp Mid Term Project
# 

# ## Importing libraries and dependencies:

import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from scipy import stats
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import export_text
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

# from tqdm.auto import tqdm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics


import warnings

warnings.filterwarnings("ignore")
SEED = 1
np.random.seed(SEED)

print(f"pandas version  : {pd.__version__}")
print(f"numpy version   : {np.__version__}")
print(f"seaborn version : {sns.__version__}")
print(f"scikit-learn version  : {sklearn.__version__}")

# ## Load the Data

# @ READING DATASET:
PATH = "./ENB2012_data.csv"
df = pd.read_csv(PATH)

df.head()



# ## Data Exploration


df_class = df.copy()


df_class.columns = [
    'Relative Compactness',
    'Surface Area',
    'Wall Area',
    'Roof Area',
    'Overall Height',
    'Orientation',
    'Glazing Area',
    'Glazing Area Distribution',
    'Heating Load',
    'Cooling Load',
]

df_class.head()

df_class.columns = df_class.columns.str.lower().str.replace(' ', '_')


# for col in df_class.columns:
#     print(col)
# 
# We don't have any NULL values, all the values in all the columns are present.

# # Training, splitting

# @ SPLITTING THE DATASET FOR TRAINING AND TEST:
from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df_class,
                                          test_size=0.2,
                                          random_state=1)
df_train, df_val = train_test_split(df_full_train,
                                    test_size=0.25,
                                    random_state=SEED)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = df_train[['heating_load', 'cooling_load']].values
y_val = df_val[['heating_load', 'cooling_load']].values
y_test = df_test[['heating_load', 'cooling_load']].values


del df_train['heating_load']
del df_train['cooling_load']

del df_test['heating_load']
del df_test['cooling_load']


# # Final model

# We select for our final model the xgboost one and train for the full set.

df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train[['heating_load', 'cooling_load']].values


del df_full_train['heating_load']
del df_full_train['cooling_load']

dicts_full_train = df_full_train.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)

dfulltrain = xgb.DMatrix(X_full_train,
                         label=y_full_train)

dtest = xgb.DMatrix(X_test)

eta = 0.3
xgb_params = {
    'min_child_weight': 2,
    'max_depth': 5,
    'eta': eta,
    'booster': 'gbtree',
    'base_score': 0.75,
    'objective': 'reg:squarederror',
    'nthread': 8,
    'seed': SEED,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dfulltrain, num_boost_round=200)
print("Training model...")
y_pred = model.predict(dtest)
rmse_final = np.sqrt(mean_squared_error(y_test, y_pred))
r2_score_final = r2_score(y_test, y_pred)
print(f"Xgboost with rmse= {rmse_final}, r2 score test : {r2_score_final}")


import bentoml


bentoml.xgboost.save_model(
    'energy_efficiency_model',
    model,
    custom_objects={
        'dictVectorizer': dv
    },
    signatures={
        "predict": {
            "batchable": True,
            "batch_dim": 0,
        }
    }
)

print("Save model completed")




