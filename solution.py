# %%

import pandas as pd
import numpy as np
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler,OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# %%
cnx = sqlite3.connect('fishing.db')

data = pd.read_sql_query("SELECT * FROM fishing", cnx)
cnx.close()
df=data.copy()
df

# %% [markdown]
# # Cleaning and Filling starts here

# %% [markdown]
# ### sunshine has negative data doing absolute

# %%
df['Sunshine'].value_counts()

# %%
df['Sunshine']= df['Sunshine'].abs()

# %%
df.isnull().sum()

# %% [markdown]
# ### converting date column to day, month, year - 3 different columns and then deleting date column

# %%
df['Date'] = pd.to_datetime(df['Date'])

df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Year'] = df['Date'].dt.year
del df['Date']

# %% [markdown]
# ### replacing categorical data with numerical which is ordinal

# %%
df['Pressure3pm']=df['Pressure3pm'].str.lower()
df['Pressure9am']=df['Pressure9am'].str.lower()
df['Pressure3pm'].replace({'low' : 1,'med' : 2,'high':3},inplace=True)
df['Pressure9am'].replace({'low' : 1,'med' : 2,'high':3},inplace=True)


# %% [markdown]
# checking null

# %%
df.isnull().sum()

# %% [markdown]
# checking df's info

# %%
df.info()

# %% [markdown]
# looking at df

# %%
df

# %% [markdown]
# replacing raintoday and raintomorrow column with binary classification where 0 is no and 1 is yes

# %%
df['RainToday'].replace({'No' : 0,'Yes' : 1},inplace=True)
df['RainTomorrow'].replace({'No' : 0,'Yes' : 1},inplace=True)

# %%
df

# %% [markdown]
# removing unnecessary column

# %%
del df['ColourOfBoats']

# %% [markdown]
# lets check targeted columns

# %%
df.info()

# %% [markdown]
# if rain fall is greater then 1.0,rain today=true else false...

# %%
df['RainToday'].loc[df['Rainfall']>=1.0] = 1
df['RainToday'].loc[df['Rainfall']<1.0] = 0


# %%
df['RainToday'].value_counts()

# %%
df['RainTomorrow'].value_counts()

# %% [markdown]
# ...hence, we know data is imballanced...

# %% [markdown]
# ### sort values

# %%
df.sort_values(by=['Location','Year', 'Month', 'Day'],inplace=True)

# %%
df_updated=df.copy()

# %%
print(df_updated['RainTomorrow'].value_counts())
df_updated['RainToday'].value_counts()

# %% [markdown]
# adjusting raintomorrow with rain today

# %%
# df_updated['RainToday-1'] = df_updated['RainToday'].shift(-1)
for i in range(len(df) - 1):
    if df_updated['RainToday'][i + 1] == 1:
        df_updated.at[i, 'RainTomorrow'] = 1
    else: df_updated.at[i, 'RainTomorrow'] = 0


# %%
count=0

for i in range(len(df) - 1):
    if df_updated['RainToday'][i+1] == 1 and df_updated['RainTomorrow'][i] != 1:
        count+=1
    if df_updated['RainToday'][i+1] == 0 and df_updated['RainTomorrow'][i] != 0:
        count+=1

# %%
count

# %%
df_updated

# %%
df_updated['RainTomorrow'].value_counts()

# %% [markdown]
# lets update rain data

# %%
# df=df_updated.copy()

# %% [markdown]
# ! above line is commented cause it will ruin the predictions and increase uncertainity

# %%
print(df.isnull().sum())
df.info()

# %% [markdown]
# creting df_merged for further filling and cleaning

# %%
df_merged=df.copy()

def calculate_mode(arr):
    arr_without_nan = arr.dropna()
    if len(arr_without_nan) == 0:
        return np.nan
    unique_values, counts = np.unique(arr_without_nan, return_counts=True)
    mode_index = np.argmax(counts)
    return unique_values[mode_index]


# %% [markdown]
# ## calculating mode according to the location, year and month and then filling empty data using it

# %%
mode_per_location = df_merged.groupby(['Location','Year','Month'])['WindGustDir'].apply(calculate_mode).reset_index()
df_merged=df_merged.merge(mode_per_location, on=['Location','Year','Month'], suffixes=('', '_mode'))

mode_per_location = df_merged.groupby(['Location','Year','Month'])['WindDir9am'].apply(calculate_mode).reset_index()
df_merged=df_merged.merge(mode_per_location, on=['Location','Year','Month'], suffixes=('', '_mode'))

mode_per_location = df_merged.groupby(['Location','Year','Month'])['WindDir3pm'].apply(calculate_mode).reset_index()
df_merged=df_merged.merge(mode_per_location, on=['Location','Year','Month'], suffixes=('', '_mode'))

for index, row in df_merged.iterrows():
    if pd.isnull(row['WindDir9am']):
        df_merged.at[index, 'WindDir9am'] = row['WindDir9am_mode']
    if pd.isnull(row['WindDir3pm']):
        df_merged.at[index, 'WindDir3pm'] = row['WindDir3pm_mode']
    if pd.isnull(row['WindGustDir']):
        df_merged.at[index, 'WindGustDir'] = row['WindGustDir_mode']
del df_merged['WindDir3pm_mode']
del df_merged['WindGustDir_mode']
del df_merged['WindDir9am_mode']
df_merged

# %% [markdown]
# looking for empty data in df-merged

# %%
df_merged.isnull().sum()

# %% [markdown]
# ### lets look at our data

# %%
df_merged.info()

# %% [markdown]
# ## fill missing data using rolling median

# %%
new_df=df_merged.copy()
window_size=10

n_colls=['Sunshine','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','AverageTemp']

for col in n_colls:
    rolling_median = new_df[col].rolling(window=window_size, min_periods=1).median()
    new_df[col]=new_df[col].fillna(rolling_median)

rolling_median = new_df['Evaporation'].rolling(window=20, min_periods=1).median()
new_df['Evaporation']=new_df['Evaporation'].fillna(rolling_median)

rolling_median = new_df['WindGustSpeed'].rolling(window=20, min_periods=1).median()
new_df['WindGustSpeed']=new_df['WindGustSpeed'].fillna(rolling_median)

rolling_median = new_df['Cloud3pm'].rolling(window=30, min_periods=1).median()
new_df['Cloud3pm']=new_df['Cloud3pm'].fillna(rolling_median)

rolling_median = new_df['Cloud9am'].rolling(window=20, min_periods=1).median()
new_df['Cloud9am']=new_df['Cloud9am'].fillna(rolling_median)

# %% [markdown]
# check for null data

# %%
new_df.isnull().sum()

# %% [markdown]
# convert type to int

# %%
new_df['Cloud9am']=new_df['Cloud9am'].astype(int)
new_df['Cloud3pm']=new_df['Cloud3pm'].astype(int)
new_df['RainToday']=new_df['RainToday'].astype(int)
new_df['WindSpeed3pm']=new_df['WindSpeed3pm'].astype(int)
new_df['WindSpeed9am']=new_df['WindSpeed9am'].astype(int)
new_df['Pressure9am']=new_df['Pressure9am'].astype(int)
new_df['Pressure3pm']=new_df['Pressure3pm'].astype(int)
# new_df['WindGustSpeed']=new_df['WindGustSpeed'].astype(int)

# %% [markdown]
# ## lets look at categorical aand linear data first

# %%
df=new_df.copy()
for col in df:
    if df[col].isnull().any():
        print("Skipping column '{}' due to presence of None values.".format(col))
        continue

    unique_vals = np.unique(df[col])
    num = len(unique_vals)
    if(num<11):
        print("the number of values for variable {} :{} --{}".format(col,num,unique_vals))
    else:
        print("the number of values for variable {} :{}".format(col,num))
    

# print(df['WindGustDir'])

# %% [markdown]
# ## lets visualize the some chosen data which can probably impact

# %%
sns.pairplot(df[['Rainfall','Cloud9am','Cloud3pm','AverageTemp','RainToday','RainTomorrow','Location']],hue='Location',kind="reg")

# %% [markdown]
# ## now lets see the correlation matrix for the data

# %%
numeric_cols = df.select_dtypes(include=['int', 'float']).columns

correlation_matrix = df[numeric_cols].corr()

numeric_cols.drop(['Pressure9am','Pressure3pm','Year','Month','Day'])

sns.heatmap(correlation_matrix)

# %% [markdown]
# ## boxplots for outliers detestions

# %%
sns.boxplot(df[numeric_cols].values,orient='v')


for c in numeric_cols:
    x = df[c].values
    ax = sns.boxplot(x, color = '#D1EC46',orient='v')
    print('The meadian is: ', df[c].median())
    plt.title(c)
    plt.show()

# %%
df=new_df.copy()

# %% [markdown]
# ### filling some unbelievable data with rolling mean

# %%
rolling_mean = df['Evaporation'].rolling(window=5).mean()
df['Evaporation'][df['Evaporation'] > 40] = rolling_mean[df['Evaporation'] > 40]
print(rolling_mean[df['Evaporation'] > 40])
sns.boxplot(df['Evaporation'].values)
plt.title('Evaporation')
plt.show()

# %%
rolling_mean = df['WindGustSpeed'].rolling(window=5).mean()
outlier_condition = df['WindGustSpeed'] > 120
df['WindGustSpeed'][outlier_condition] = rolling_mean[outlier_condition]
sns.boxplot(df['WindGustSpeed'].values)
plt.show()

# %%
rolling_mean = df['Rainfall'].rolling(5).mean()
df['Rainfall'][df['Rainfall']>350] = rolling_mean[df['Rainfall']>350]
print(rolling_mean[df['Rainfall']>350])
sns.boxplot(df['Rainfall'].values)
plt.show()

# %% [markdown]
# check for null data

# %%
df.isnull().sum()

# %% [markdown]
# categorizing the wind direction column

# %%
def encode_wind_directions(df, column_name):
    # Create the direction columns
    direction_columns = [f'{column_name}_N', f'{column_name}_S', f'{column_name}_E', f'{column_name}_W']
    for column in direction_columns:
        df[column] = 0

    # Iterate through the DataFrame and update direction columns
    for index, row in df.iterrows():
        wind_direction = row[column_name]
        if isinstance(wind_direction, str):
            for char in wind_direction:
                if char == 'N':
                    df.at[index, f'{column_name}_N'] += 1
                elif char == 'S':
                    df.at[index, f'{column_name}_S'] += 1
                elif char == 'E':
                    df.at[index, f'{column_name}_E'] += 1
                elif char == 'W':
                    df.at[index, f'{column_name}_W'] += 1

    # Drop the original column
    df.drop(columns=[column_name], inplace=True)

# %%
temp_df=df.copy()
for cols in 'WindGustDir','WindDir9am','WindDir3pm':
    encode_wind_directions(temp_df,cols)


# %% [markdown]
# ### one hottie encoding...

# %%
encoder = OneHotEncoder()
temp_df['Location'] = temp_df['Location'].astype('category')
one_hot_encoded = pd.get_dummies(temp_df['Location'], prefix='Location')
temp_df = pd.concat([temp_df, one_hot_encoded], axis=1)
temp_df.drop(columns=['Location'], inplace=True)

temp_df

# %%
df=temp_df.copy()

mscaler = MinMaxScaler()  # Or: StandardScaler()
sscaler=StandardScaler()

StandardScaler_scaled_data = sscaler.fit_transform(df.drop(['RainTomorrow'],axis=1))
Min_Max_scaled_data = mscaler.fit_transform(df.drop(['RainTomorrow'],axis=1))

# %% [markdown]
# ## Correlation Matrix

# %%

correlation_matrix = df.corr()

print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# %% [markdown]
# we cant understand anything from it so lets go further and look at feature importance

# %% [markdown]
# ## Looking At Feature Importance through Random Forest Classifier

# %%
from sklearn.ensemble import RandomForestClassifier

# Load your data and split it into features (X) and target (y)
X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']

# Initialize the random forest classifier
rf_classifier = RandomForestClassifier()

# Fit the model to your data
rf_classifier.fit(X, y)

# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort and display the top features
top_features = feature_importance_df.sort_values(by='Importance', ascending=False)
print(top_features)

# %% [markdown]
# looks like most features are important so we cant avoid them...
# so lets just not do feature elimination

# %% [markdown]
# # Models:-

# %% [markdown]
# ## Train Test Splitting

# %%
from sklearn.metrics import accuracy_score

X = df.drop(['RainTomorrow'], axis=1)
y = df['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)


# %% [markdown]
# ## Logistic Regression

# %%


from sklearn.linear_model import LogisticRegression


logistic_regression_model = LogisticRegression(max_iter=10000)
# logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)



y_pred = logistic_regression_model.predict(X_test)

score = logistic_regression_model.score(X_test, y_test)
print("Accuracy: {:.2f}".format(score*100))
print("Accuracy of Logistic Regression:",accuracy_score(y_test, y_pred))

# %% [markdown]
# ## DecisionTree Classifier
# 
# ### gini

# %%

decision_tree_model = DecisionTreeClassifier(random_state=15, criterion  = 'gini', max_depth = 14)
decision_tree_model.fit(X_train, y_train)

y_pred_decision_tree_model = decision_tree_model.predict(X_test)

score = decision_tree_model.score(X_test, y_test)
print("Accuracy: {:.2f}".format(score*100))
print("Accuracy of Decision tree with gini:",accuracy_score(y_test, y_pred_decision_tree_model))


# %% [markdown]
# ### Entropy

# %%
from sklearn import tree


dt = DecisionTreeClassifier(random_state=15, criterion  = 'entropy', max_depth = 13)

dt.fit(X,y)

y_pred_dt = dt.predict(X_test)

score = dt.score(X_test, y_test)
print("Accuracy: {:.2f}".format(score*100))
print("Accuracy of Decision tree with entropy:",accuracy_score(y_test, y_pred_dt))
fig = plt.figure(figsize=(25,20))
# tree.plot_tree(dt, 
#                    feature_names=X.columns,  
#                    class_names=y.name,
#                    filled=True)


# %% [markdown]
# ### log_loss

# %%

dtl = DecisionTreeClassifier(random_state=15, criterion  = 'log_loss',min_samples_split=8)
dtl.fit(X,y)
y_pred_dtl = dtl.predict(X_test)

score = dtl.score(X_test, y_test)
print("Accuracy: {:.2f}".format(score*100))
print("Accuracy of Decision tree with log_loss:",accuracy_score(y_test, y_pred_dtl))


# %% [markdown]
# ## Random Forest Model
# 
# ### Gini

# %%

random_forest_model = RandomForestClassifier(random_state=15, criterion  = 'gini',max_depth=14)
random_forest_model.fit(X_train, y_train)

y_pred = random_forest_model.predict(X_test)

score = random_forest_model.score(X_test, y_test)
print("Accuracy: {:.2f}".format(score*100))
print("Accuracy of Random Forest Classifier:",accuracy_score(y_test, y_pred))



# %% [markdown]
# ### Entropy

# %%

random_forest_model = RandomForestClassifier(random_state=15, criterion  = 'entropy',max_depth=13)
random_forest_model.fit(X_train, y_train)

y_pred = random_forest_model.predict(X_test)

score = random_forest_model.score(X_test, y_test)
print("Accuracy: {:.2f}".format(score*100))
print("Accuracy of Random Forest Classifier:",accuracy_score(y_test, y_pred))



# %% [markdown]
# ### log_loss

# %%

random_forest_model = RandomForestClassifier(random_state=15, criterion  = 'log_loss',min_samples_split=8)
random_forest_model.fit(X_train, y_train)

y_pred = random_forest_model.predict(X_test)

score = random_forest_model.score(X_test, y_test)
print("Accuracy: {:.2f}".format(score*100))
print("Accuracy of Random Forest Classifier:",accuracy_score(y_test, y_pred))

# %% [markdown]
# ## XGBoost, LightGBM, CatBoost

# %%
import xgboost
import lightgbm as lgb
import catboost as cb

xgb_model = xgboost.XGBClassifier()
xgb_model.fit(X_train, y_train)

# Create and fit the LightGBM model
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)

# Create and fit the CatBoost model
cb_model = cb.CatBoostClassifier()
cb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_xgb = xgb_model.predict(X_test)
y_pred_lgb = lgb_model.predict(X_test)
y_pred_cb = cb_model.predict(X_test)

score = xgb_model.score(X_test, y_test)
print("Accuracy of xgb_model: {:.2f}".format(score*100))
print("Accuracy of xgb_model:",accuracy_score(y_test, y_pred_xgb))
score = lgb_model.score(X_test, y_test)
print("Accuracy of lgb_model: {:.2f}".format(score*100))
print("Accuracy of lgb_model:",accuracy_score(y_test, y_pred_lgb))
score = cb_model.score(X_test, y_test)
print("Accuracy of cb_model: {:.2f}".format(score*100))
print("Accuracy of cb_model:",accuracy_score(y_test, y_pred_cb))

# %% [markdown]
# ## Support Vector Classifier

# %%
from sklearn.svm import SVC


svm_model = SVC(kernel="linear") # tried all but linear gives best output.
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

score = svm_model.score(X_test, y_test)
print("Accuracy of svm_model: {:.2f}".format(score*100))
print("Accuracy of svm:{:.2f}".format(accuracy_score(y_test, y_pred)*100))

# %% [markdown]
# ## Gradient Boost

# %%
from sklearn.ensemble import GradientBoostingClassifier


gbc = GradientBoostingClassifier(random_state=0)
gbc.fit(X_train, y_train)

y_pred = gbc.predict(X_train)

print('The Accuracy  on the testing dataset is: {:.2f}'.format(gbc.score(X_test, y_test)*100))


# %% [markdown]
# ## Neural Network

# %%

neural_network_model = Sequential()

input_dim = X_train.shape[1]

# Add layers to the model
neural_network_model.add(Dense(64, activation='relu', input_dim=input_dim))
neural_network_model.add(Dense(64, activation='relu'))
neural_network_model.add(Dense(1, activation='sigmoid'))

# Compile the model
neural_network_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model to the training data
neural_network_model.fit(X_train, y_train, epochs=50, batch_size=46)

# Predict on the test set
y_pred_prob = neural_network_model.predict(X_test)
classes_x=np.argmax(y_pred)
y_pred = (y_pred_prob > 0.5).astype(int)


# %%
print("Accuracy of neural_network_model:{:.2f}".format(accuracy_score(y_test, y_pred)*100))

# %% [markdown]
# ## Naive Bayes Gaussian and Bernaulli with KFold

# %%
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
from sklearn.naive_bayes import BernoulliNB as bnb

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=16)

kf = KFold(n_splits=40)
acc_gnb=[]
acc_bnb=[]
for i, (train_index, test_index) in enumerate(kf.split(X_train)):
    # print("---------------------------------------------------------------------------------")
    # print(f"Fold {i}:")
    # print(f"  Train: index={train_index}")
    # print(f"  Test:  index={test_index}")
    current_X_train = X_train.iloc[train_index]
    current_Y_train = Y_train.iloc[train_index]
    current_X_test = X_train.iloc[train_index]
    current_Y_test = Y_train.iloc[train_index]
    gnb_model = GaussianNB()
    gnb_model = gnb_model.fit(current_X_train, current_Y_train)
    current_Y_pred_gnb=gnb_model.predict(current_X_test)
    acc_gnb.insert(i,metrics.accuracy_score(current_Y_test,current_Y_pred_gnb))
    bnb_model = bnb()
    bnb_model.fit(current_X_train, current_Y_train)
    current_Y_pred_bnb = bnb_model.fit(current_X_train, current_Y_train).predict(current_X_test)
    # print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (Y_test != current_Y_pred_bnb).sum()))
    acc_bnb.insert(i,metrics.accuracy_score(current_Y_test,current_Y_pred_bnb))
    # print(acc_gnb[i])
    # print(acc_bnb[i])

print("gaussian",np.mean(acc_gnb,axis=0)*100)
print('burnouli',np.mean(acc_bnb,axis=0)*100)

# %% [markdown]
# # Conclusion:
# 
# Logistic regression,Random Forest,Gradient Boosting Classifier, X Gradient Boost Model, L Gradient Boost Model and Catboost are the ones giving above 85% accuracy...
# - Logistic regression: 86.38%
# - Decision Tree Classifier:-
#     - gini: 83.31%
#     - Entropy: 95.77%
#     - log_loss: 96.92%
# - Random Forest:-
#     - gini: 88.31%
#     - Entropy: 89.65%
#     - log_loss: 88.54%
# - X Gradient Boost Model : 88.04%
# - Light Gradient Boost Model : 87.35%
# - Cat Boost Model : 88.46%
# - Gradient Boost Model : 87.54%
# - Support Vector Classifier: 86.00%
# - Neural Network : 86.42%
# 
# So here we conclude this Problem eith the log_loss as the winner for nearly 97% accuracy score


