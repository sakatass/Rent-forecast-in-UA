#!/usr/bin/env python
# coding: utf-8

# ## Load libraries

# In[1]:


import pandas as pd
import requests
import os
import warnings
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import requests 
import os 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, cv


# ## Dataset Creation

# In[2]:


def create_directory(directory_path):
    # Проверяем, существует ли директория
    if not os.path.exists(directory_path):
        # Создаем директорию, если она не существует
        os.makedirs(directory_path)
        print(f"Директория {directory_path} успешно создана.")
    else:
        print(f"Директория {directory_path} уже существует.")

# Пример использования:
directory_path = "HousePrice_excels"
create_directory(directory_path)


# ### Download excel files

# In[3]:


def download_excel_selenim(download_directory, n_page, state_id):
    files_count = os.listdir(download_directory)
    # Set up ChromeOptions
    chrome_options = webdriver.ChromeOptions()

    # Set the download directory
    prefs = {
       "download.default_directory": download_directory,
       "savefile.default_directory": download_directory
    }
    chrome_options.add_experimental_option('prefs', prefs)
    #chrome_options.add_argument('--headless=new')

    # Instantiate the Chrome driver with the configured options
    driver = webdriver.Chrome(ChromeDriverManager().install(),
                              chrome_options=chrome_options
                             )
    driver.get(f'https://dom.ria.com/uk/search/xls/?excludeSold=1&category=1&realty_type=2&operation=3&state_id={state_id}&price_cur=1&wo_dupl=1&sort=inspected_sort&firstIteraction=false&client=searchV2&limit=20&page={n_page}&type=list&ch=246_244&xls=1')
    time.sleep(1.2)
    if os.listdir(download_directory) == files_count:
        input()
    else:
        pass


# In[5]:


for state_id in range(1, 25):
    for n_page in range(1, 30):
        download_excel_selenim(os.path.join(os.getcwd(), directory_path), state_id, n_page)


# In[ ]:


for state_id in range(21, 25):
    for n_page in range(1, 30):
        download_excel_selenim(os.path.join(os.getcwd(), directory_path), state_id, n_page)


# ### Combine excel files to a DataFrame

# In[6]:


combined_df = pd.DataFrame()

for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    try:
        df = pd.read_excel(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    except Exception as e:
        print(f"An error occurred while reading a file {file_path}: {e}")


# In[7]:


combined_df.info()


# ## Dataset Preprocessing

# In[8]:


combined_df.drop_duplicates(inplace=True)


# ### Drop useless columns

# In[9]:


combined_df.columns = df.columns.str.strip()

for col in combined_df.columns:
    if combined_df[col].isna().sum() > combined_df.shape[0] * 0.5:
        combined_df.drop(col, axis=1, inplace=True)
        
useless_columns = ['Realty ID',
                  'Дата',
                  'Агенція',
                  'Користувач',
                  'Опис',
                  'Тип нерухомості'
                  ]
combined_df.drop(useless_columns, axis=1, inplace=True)


# In[10]:


combined_df.head(2)


# ### Change columns dtype

# In[11]:


int_cols = ['Ціна', 'Кімнат', 'Поверх', 'Поверховість']
float_cols = ['Загальна площа', 'Житлова площа', 'Кухня']

for col_name in int_cols:
    combined_df[col_name] = combined_df[col_name].astype(int)
for col_name in float_cols:
    combined_df[col_name] = combined_df[col_name].apply(lambda x: str(x).replace(',', '.')).astype(float)


# ### Preprocessing Price column

# In[12]:


combined_df['Тип ціни'].unique()


# In[13]:


price_ratio = {'грн': 1,
                '$': 37,
                '€': 40,}

combined_df['Ціна'] = combined_df \
        .apply(lambda x: x['Ціна'] * price_ratio[x['Тип ціни']], axis=1)


# In[14]:


combined_df.drop('Тип ціни', axis=1, inplace=True)


# ### Drop outliers using Z-score

# In[15]:


num_cols = combined_df.select_dtypes(exclude=object).columns

for col_name in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=combined_df[col_name])
    plt.title(f'Column name: {col_name}')
    plt.show()


# In[16]:


for col_name in num_cols:
    z_scores = zscore(combined_df[col_name])
    z_score_threshold = 3
    outliers_mask = (abs(z_scores) > z_score_threshold)
    combined_df = combined_df[~outliers_mask]
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=combined_df[col_name])
    plt.title(f'Column name: {col_name}')
    plt.show()


# In[17]:


combined_df.info()


# ### Fill NaN-value

# In[18]:


missing_values = combined_df.isnull().sum()

percent_missing = (missing_values / len(combined_df)) * 100

missing_info = pd.DataFrame({'Missing Values': missing_values, 'Percent Missing': percent_missing})

print("Информация о пропущенных значениях:")
print(missing_info)

plt.figure(figsize=(10, 6))
sns.barplot(x=missing_info.index, y='Percent Missing', data=missing_info)
plt.title('Процент пропущенных значений по колонкам')
plt.xlabel('Колонки')
plt.ylabel('Процент пропущенных значений')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[19]:


mode_value = combined_df['Район'].mode().iloc[0]
combined_df['Район'] = combined_df['Район'].fillna(value=mode_value)


# In[20]:


mode_value = combined_df['Вулиця'].mode().iloc[0]
combined_df['Вулиця'] = combined_df['Вулиця'].fillna(value=mode_value)

mode_value = combined_df['Стіни'].mode().iloc[0]
combined_df['Стіни'] = combined_df['Стіни'].fillna(value=mode_value)


# #### Fill 'Житлова площа' & 'Кухня'

# In[21]:


livArea_kitchen_df = combined_df[['Кімнат', 'Загальна площа', 'Житлова площа', 'Кухня']]
livArea_kitchen_df.dropna(inplace=True)
#livArea_kitchen_df = livArea_kitchen_df[livArea_kitchen_df['Кухня'] < 60]


# In[22]:


livArea_kitchen_df.info()


# #### Models training

# In[24]:


import math
X = livArea_kitchen_df.drop(['Житлова площа', 'Кухня'], axis=1)
y = livArea_kitchen_df['Житлова площа']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_lArea = LinearRegression()
model_lArea.fit(X_train, y_train)
predictions = model_lArea.predict(X_test)

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


print(f'lArea Root Mean Squared Error: {math.sqrt(mse)}')
print(f'lArea Mean Absolute Error: {mae}')
print(f'lArea R-squared: {r2}')
print('')


X = livArea_kitchen_df.drop(['Житлова площа', 'Кухня'], axis=1)
y = livArea_kitchen_df['Кухня']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_kitchen = LinearRegression()
model_kitchen.fit(X_train, y_train)
predictions = model_kitchen.predict(X_test)

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'kitchen Root Mean Squared Error: {math.sqrt(mse)}')
print(f'kitchen Mean Absolute Error: {mae}')
print(f'kitchen R-squared: {r2}')


# In[25]:


print(model_lArea.feature_names_in_)
print(model_kitchen.feature_names_in_)


# In[26]:


combined_df_V2 = combined_df.copy()


# In[ ]:





# #### Fill

# In[27]:


column_with_nan = 'Житлова площа'

rows_with_nan = combined_df_V2[combined_df_V2[column_with_nan].isna()]

X_for_prediction = rows_with_nan[model_lArea.feature_names_in_]

predictions_lArea = model_lArea.predict(X_for_prediction)
rows_with_nan[column_with_nan] = predictions_lArea

combined_df_V2 = pd.concat([combined_df_V2.drop('Кухня', axis=1).dropna(), rows_with_nan])



column_with_nan = 'Кухня'

rows_with_nan = combined_df_V2[combined_df_V2[column_with_nan].isna()]

X_for_prediction = rows_with_nan[model_lArea.feature_names_in_]

predictions_lArea = model_lArea.predict(X_for_prediction)
rows_with_nan[column_with_nan] = predictions_lArea

combined_df_V2 = pd.concat([combined_df_V2.dropna(), rows_with_nan])


# In[28]:


missing_values = combined_df_V2.isnull().sum()

percent_missing = (missing_values / len(combined_df_V2)) * 100

missing_info = pd.DataFrame({'Missing Values': missing_values, 'Percent Missing': percent_missing})

print("Информация о пропущенных значениях:")
print(missing_info)

plt.figure(figsize=(10, 6))
sns.barplot(x=missing_info.index, y='Percent Missing', data=missing_info)
plt.title('Процент пропущенных значений по колонкам')
plt.xlabel('Колонки')
plt.ylabel('Процент пропущенных значений')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[29]:


combined_df.to_csv('HousePrice_df_versions/combined_df.csv', index=False)
combined_df_V2.to_csv('HousePrice_df_versions/combined_df_V2.csv', index=False)


# In[30]:


combined_df_V2 = pd.read_csv('C:/Users/MLZZZDS/HousePrice_df_versions/combined_df_V2.csv')


# In[31]:


combined_df_V3 = pd.get_dummies(combined_df_V2)


# ## Models Price prediction tuning

# In[32]:


combined_df_V3.shape


# In[33]:


X = combined_df_V3.drop('Ціна', axis=1)
X.columns = X.columns.str.replace('[^a-zA-Z0-9А-Яа-яіїЇІЄє]', '_')
y = combined_df_V3['Ціна']


# In[34]:


'Місто_Київ' in X.columns


# #### RandomForestRegressor

# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

predictions = rf_model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Price Root Mean Squared Error: {math.sqrt(mse)}')
print(f'Price Mean Absolute Error: {mae}')
print(f'Price R-squared: {r2}')


# In[36]:


train_pred = rf_model.predict(X_train)


# In[37]:


plt.figure(figsize=(10,6))

plt.scatter(train_pred,train_pred - y_train,
          c = 'black', marker = 'o', s = 35, alpha = 0.5,
          label = 'Train data')
plt.scatter(predictions,predictions - y_test,
          c = 'c', marker = 'o', s = 35, alpha = 0.7,
          label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Tailings')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = 0, xmax = 60000, lw = 2, color = 'red')
plt.show()


# #### xgb

# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание объекта DMatrix для эффективной работы с данными в XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Определение параметров модели
params = {
    'objective': 'reg:squarederror',  # Функция потерь для задачи регрессии
    'eval_metric': 'rmse',  # Метрика оценки качества (среднеквадратичное отклонение)
    'max_depth': 3,  # Максимальная глубина дерева
    'learning_rate': 0.1,  # Скорость обучения (шаг градиентного спуска)
    'n_estimators': 300  # Количество деревьев в ансамбле
}

# Обучение модели
model = xgb.train(params, dtrain, num_boost_round=params['n_estimators'])

# Предсказание на тестовом наборе
predictions = model.predict(dtest)

# Оценка модели
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Price Root Mean Squared Error: {math.sqrt(mse)}')
print(f'Price Mean Absolute Error: {mae}')
print(f'Price R-squared: {r2}')


# ##### Tuning params

# In[39]:


get_ipython().run_cell_magic('time', '', '\nparam_grid = {\n    \'max_depth\': [3, 5, 7, 9, 11],\n    \'learning_rate\': [0.01, 0.1, 0.2, 0.5, 1],\n    \'n_estimators\': [100, 300, 500, 700],\n    \'tree_method\': [\'hist\']\n}\n\n# Создание модели XGBRegressor\nxgb_model = xgb.XGBRegressor(objective=\'reg:squarederror\')\n\n# Определение метрики для оценки\nscorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)\n\n# Создание объекта GridSearchCV\ngrid_search = GridSearchCV(\n    xgb_model,\n    param_grid,\n    scoring=scorer,\n    cv=KFold(n_splits=5, shuffle=True, random_state=42),  # Пример использования 5-кратной кросс-валидации\n    verbose=1,\n    n_jobs=-1  # Используйте все доступные ядра процессора\n)\n\n# Выполнение поиска по сетке\ngrid_result = grid_search.fit(X_train, y_train)\n\n# Вывод лучших параметров и оценки\nprint("Best Parameters: ", grid_result.best_params_)\nprint("Best RMSE: ", math.sqrt(-grid_result.best_score_))\n\n# Обучение модели с лучшими параметрами\nbest_model = grid_result.best_estimator_\nbest_model.fit(X_train, y_train)\n\n# Предсказание на тестовом наборе\npredictions = best_model.predict(X_test)\n\n# Оценка модели\nmse = mean_squared_error(y_test, predictions)\nmae = mean_absolute_error(y_test, predictions)\nr2 = r2_score(y_test, predictions)\n\nprint(f\'Price Root Mean Squared Error: {math.sqrt(mse)}\')\nprint(f\'Price Mean Absolute Error: {mae}\')\nprint(f\'Price R-squared: {r2}\')\n')


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                            learning_rate=0.2, max_depth=3, n_estimators=500, tree_method='hist')
xgb_model.fit(X_train, y_train)
predictions = xgb_model.predict(X_test)
# Оценка модели
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Price Root Mean Squared Error: {math.sqrt(mse)}')
print(f'Price Mean Absolute Error: {mae}')
print(f'Price R-squared: {r2}')


# In[41]:


joblib.dump(xgb_model, 'xgboost_model.pkl')


# In[42]:


'Місто_Київ' in xgb_model.feature_names_in_


# In[44]:


import pickle
with open('xgboost_model.pkl', 'wb') as model_file:
    pickle.dump(xgb_model, model_file)


# In[45]:


import pickle

with open('xgboost_model.pkl', 'rb') as model_file:
    xgbr_model = pickle.load(model_file)


# In[ ]:





# In[ ]:





# In[48]:


get_ipython().run_cell_magic('time', '', '\nfrom sklearn.metrics import make_scorer\n\n\nparam_grid = {\n    \'max_depth\': [3, 5],\n    \'learning_rate\': [0.001, 0.01, 0.1, 0.2],\n    \'n_estimators\': [100, 300, 500, 700],\n    \'tree_method\': [\'hist\'],\n}\n\n# Создание модели XGBRegressor\nxgb_model = xgb.XGBRegressor(objective=\'reg:squarederror\')\n\n# Определение метрики для оценки\nscorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)\n\n# Создание объекта GridSearchCV\ngrid_search = GridSearchCV(\n    xgb_model,\n    param_grid,\n    scoring=scorer,\n    cv=KFold(n_splits=5, shuffle=True, random_state=42),  # Пример использования 5-кратной кросс-валидации\n    verbose=1,\n    n_jobs=-1  # Используйте все доступные ядра процессора\n)\n\n# Выполнение поиска по сетке\ngrid_result = grid_search.fit(X_train, y_train)\n\n# Вывод лучших параметров и оценки\nprint("Best Parameters: ", grid_result.best_params_)\nprint("Best RMSE: ", math.sqrt(-grid_result.best_score_))\n\n# Обучение модели с лучшими параметрами\nbest_model = grid_result.best_estimator_\nbest_model.fit(X_train, y_train)\n\n# Предсказание на тестовом наборе\npredictions = best_model.predict(X_test)\n\n# Оценка модели\nmse = mean_squared_error(y_test, predictions)\nmae = mean_absolute_error(y_test, predictions)\nr2 = r2_score(y_test, predictions)\n\nprint(f\'Price Root Mean Squared Error: {math.sqrt(mse)}\')\nprint(f\'Price Mean Absolute Error: {mae}\')\nprint(f\'Price R-squared: {r2}\')\n')


# In[ ]:





# #### LGBM

# In[49]:


model = LGBMRegressor(
    objective='regression',  
    max_depth=3, 
    learning_rate=0.1,  
    n_estimators=300  
)

# Обучение модели
model.fit(X_train, y_train)

# Предсказание на тестовом наборе
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Price Root Mean Squared Error: {math.sqrt(mse)}')
print(f'Price Mean Absolute Error: {mae}')
print(f'Price R-squared: {r2}')


# ##### Tuning params

# In[50]:


get_ipython().run_cell_magic('time', '', '\nparam_grid = {\n    \'max_depth\': [3, 5, 7, 9, 11],\n    \'learning_rate\': [0.01, 0.1, 0.2, 0.5, 1],\n    \'n_estimators\': [100, 300, 500, 700],\n    \'reg_alpha\': [0.01, 0.1, 1, 10, 100],\n    \'reg_lambda\': [0.01, 0.1, 1, 10, 100]\n}\n\n\nmodel = LGBMRegressor(\n    objective=\'regression\',  \n)\n\n# Определение метрики для оценки\nscorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)\n\n# Создание объекта GridSearchCV\ngrid_search = GridSearchCV(\n    model,\n    param_grid,\n    scoring=scorer,\n    cv=KFold(n_splits=5, shuffle=True, random_state=42),  # Пример использования 5-кратной кросс-валидации\n    verbose=1,\n    n_jobs=-1  # Используйте все доступные ядра процессора\n)\n\n# Выполнение поиска по сетке\ngrid_result = grid_search.fit(X_train, y_train)\n\n# Вывод лучших параметров и оценки\nprint("Best Parameters: ", grid_result.best_params_)\nprint("Best RMSE: ", math.sqrt(-grid_result.best_score_))\n\n# Обучение модели с лучшими параметрами\nbest_model = grid_result.best_estimator_\nbest_model.fit(X_train, y_train)\n\n# Предсказание на тестовом наборе\npredictions = best_model.predict(X_test)\n\n# Оценка модели\nmse = mean_squared_error(y_test, predictions)\nmae = mean_absolute_error(y_test, predictions)\nr2 = r2_score(y_test, predictions)\n\nprint(f\'Price Root Mean Squared Error: {math.sqrt(mse)}\')\nprint(f\'Price Mean Absolute Error: {mae}\')\nprint(f\'Price R-squared: {r2}\')\n')


# #### DNN

# In[62]:


from keras import models, layers

# Function to create the DNN model
def create_dnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.7))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))  

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model

# Load the data
combined_df_V2 = pd.read_csv('HousePrice_df_versions/combined_df_V2.csv')
combined_df_V3 = pd.get_dummies(combined_df_V2)

X = combined_df_V3.drop('Ціна', axis=1)
X.columns = X.columns.str.replace('[^a-zA-Z0-9А-Яа-яіІЄє]', '_')
y = combined_df_V3['Ціна']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Create DNN model
dnn_model = create_dnn_model(X_train.shape[1])

# Train the model and get the history
history = dnn_model.fit(X_train, y_train, epochs=400, batch_size=128, validation_data=(X_val, y_val), verbose=0)

# Plot MSE over training epochs
plt.plot(history.history['mse'], label='Training MSE')
plt.plot(history.history['val_mse'], label='Validation MSE')
plt.title('Training and Validation MSE over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Evaluate the model on the test set
dnn_predictions = dnn_model.predict(X_test)
dnn_mse = mean_squared_error(y_test, dnn_predictions)
dnn_r2 = r2_score(y_test, dnn_predictions)

print(f'DNN Mean Squared Error: {dnn_mse}')
print(f'DNN Root Mean Squared Error: {math.sqrt(dnn_mse)}')
print(f'DNN R-squared: {dnn_r2}')


# In[63]:


# Function to create the DNN model
def create_dnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(24, activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1, activation='linear'))  

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model

# Load the data
combined_df_V2 = pd.read_csv('HousePrice_df_versions/combined_df_V2.csv')
combined_df_V3 = pd.get_dummies(combined_df_V2)

X = combined_df_V3.drop('Ціна', axis=1)
X.columns = X.columns.str.replace('[^a-zA-Z0-9А-Яа-яіІЄє]', '_')
y = combined_df_V3['Ціна']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Create DNN model
dnn_model = create_dnn_model(X_train.shape[1])

# Train the model and get the history
history = dnn_model.fit(X_train, y_train, epochs=400, batch_size=128, validation_data=(X_val, y_val), verbose=0)

# Plot MSE over training epochs
plt.plot(history.history['mse'], label='Training MSE')
plt.plot(history.history['val_mse'], label='Validation MSE')
plt.title('Training and Validation MSE over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Evaluate the model on the test set
dnn_predictions = dnn_model.predict(X_test)
dnn_mse = mean_squared_error(y_test, dnn_predictions)
dnn_r2 = r2_score(y_test, dnn_predictions)

print(f'DNN Mean Squared Error: {dnn_mse}')
print(f'DNN Root Mean Squared Error: {math.sqrt(dnn_mse)}')
print(f'DNN R-squared: {dnn_r2}')


# In[ ]:





# In[ ]:





# In[ ]:




