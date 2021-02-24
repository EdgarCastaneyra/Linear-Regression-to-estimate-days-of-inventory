import pandas as pd
import pandas.io.sql as sqlio
import psycopg2
import scipy
import statsmodels.api as sm
import os
import matplotlib.pyplot as plt  ## import fundamental plotting library in Python
import seaborn as sns  ## Advanced plotting functionality with seaborn
import numpy as np  ## Import the NumPy package

from datetime import datetime, date, time, timedelta
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

sns.set(style="whitegrid") # can set style depending on how you'd like it to look

print("Starting...")
## Conect postgres to python
conn = psycopg2.connect(database="xxxx", user = "xxxxx", password = "xxxxxx", host = "xxxxx",port="xxxxx")
cursor1=conn.cursor()


print("Saving Query...")
# Save Query into a dataframe

car_pageviews = pd.read_sql_query("select * from inv_dist.car_pageviews", conn)
showroom_visits = pd.read_sql_query("select a.*, b.name location_name from inv_dist.showroom_visits a LEFT JOIN inv_dist.showroom_id b on a.showroom_id=b.showroom_id", conn)  
showroom = pd.read_sql_query("select * from inv_dist.showroom_id", conn)
car_description = pd.read_sql_query(
    '''SELECT c.*, d.name brand_name, left(c.version,1) doors FROM
            (SELECT a.*, b.name location_name 
             FROM inv_dist.car_description a LEFT JOIN inv_dist.showroom_id b 
        ON a.sold_location_id=b.showroom_id) c
        LEFT JOIN inv_dist.brands d ON c.brand_id=d.brand_id
    ''', conn)


print("Starting Data Cleaning...")
### Add stock_ID to the dataset for Pageviews table
car_pageviews['stock_id'] = car_pageviews['gmb_pagepath'].apply(lambda x: x[(x.rfind("autos-")+6):len(x)])

##Create date columns
car_pageviews['gmb_datetime'] = pd.to_datetime(car_pageviews['gmb_date'], format="%Y-%m-%d")
car_pageviews['year'] = car_pageviews['gmb_datetime'].dt.year
car_pageviews['month'] = car_pageviews['gmb_datetime'].dt.month
car_pageviews['my'] =pd.to_datetime(("01" + "/" + car_pageviews['month'].astype(str) + "/" + car_pageviews['year'].astype(str)), format="%d/%m/%Y")


import datetime

##Delete showroom equal to "FUERA DE KAVAK"
#car_description = car_description.drop(car_description[car_description['location_name']=="FUERA DE KAVAK"].index)
list_of_values = ['otro']
car_description = car_description.drop(car_description[car_description['location_name']=='otro'].index)


## Create new columns for numeric KAVAK SEGMENTS

a=('[ 79,999 - 209,999 ]','[ 212,999 - 339,999 ]','[ 342,999 - 1054,999 ]')
a=('[ 1,050 - 36,300 ]','[ 36,649 - 64,851 ]','[ 64,965 - 110,900 ]')
a=('Low','Mid','High')

car_description['new_price_segment'] = car_description['price_segment'].map({1: "[ 79,999 - 209,999 ]", 2: "[ 212,999 - 339,999 ]", 3: "[ 342,999 - 1054,999 ]"})
car_description['new_km_segment'] = car_description['km_segment'].map({1: "[ 1,050 - 36,300 ]", 2: "[ 36,649 - 64,851 ]", 3: "[ 64,965 - 110,900 ]"})
car_description['new_model_segment'] = car_description['model_segment'].map({1: "Low", 2: "Mid", 3: "High"})


##Create date columns
car_description['sold_datetime'] = pd.to_datetime(car_description['sold_date'], format="%Y-%m-%d")
car_description['sold_year'] = car_description['sold_datetime'].dt.year
car_description['sold_month'] = car_description['sold_datetime'].dt.month
car_description['sold_my'] =pd.to_datetime(("01" + "/" + car_description['sold_month'].astype(str) + "/" + car_description['sold_year'].astype(str)), format="%d/%m/%Y")


### Sum total pageviews per car ###

car_description['tot_pageviews']=9999

for indice_fila, fila in car_description.iterrows():
    stockid=car_description.loc[indice_fila,'stock_id']
    stockid=str(stockid)
    lastdate=car_description.loc[indice_fila,'sold_datetime']
    invdays=car_description.loc[indice_fila,'inventory_days']
    invdays=int(invdays)
    firstdate = lastdate-datetime.timedelta(days=invdays)
    #if invdays>=5:
    #    invdays=5
    #else:
    #    invdays
    newlastday = firstdate+datetime.timedelta(days=6)
    pv = car_pageviews[(car_pageviews['stock_id']==stockid) & (car_pageviews['gmb_datetime']<=newlastday)]
    value=pv['views'].sum()
    car_description.loc[indice_fila,'tot_pageviews']=value

    
### cars without pageviews ###
car_description['wopv']=99

for indice_fila, fila in car_description.iterrows():
    a=int(car_description.loc[indice_fila,'tot_pageviews'])
    if a>0:
        car_description.loc[indice_fila,'wopv']=1
    else:
        car_description.loc[indice_fila,'wopv']=0
        
        
### Replace None values in doors with 4 ###

car_description[car_description.doors.isnull()]
car_description['doors'].replace(to_replace=[None], value=4, inplace=True)


## histogram plot of price
#sns.distplot(houses['price'],fit=stats.laplace, kde=False)
sns.distplot(car_description['inventory_days'],fit=stats.norm, kde=False)
sns.set(color_codes=True)
plt.xticks(rotation=90)
plt.title("Histogram of inventory days")

## QQ plot of inventroy days
plt.subplot(2,1,1)
stats.probplot(np.log(car_description['inventory_days']), dist = "norm", plot = plt)
plt.title("QQ Plot for Log inventory days")
plt.show()
plt.subplot(2,1,2)
sns.distplot(np.log(car_description['inventory_days']),fit=stats.norm, kde=False)
sns.set(color_codes=True)
plt.xticks(rotation=90)
plt.title("Histogram of Log Inventory days")

## RMSE

def RMSE(prediction,true_values):
    
    return np.sqrt(                                                          # Root
            np.mean(                                                      # Mean
                np.square(                                                # Squared
                         prediction-true_values                           # Error
                )
            )
        )
 
## MAPE
def MAPE(prediction,true_value):
    return np.mean(                                           # Mean
        np.abs(                                               # Absolute
               (prediction-true_value)/true_value             # Error
            )*100                                            # Percentage
    )

print("Running MODEL 1...")
   
#############################
####### MODEL 1 #############
#############################

# Subset Data for model1

car_description_reg=car_description[['inventory_days','km', 'published_price','tot_pageviews','year', ##Numerical variables
                                     'location_name', 'brand_name', 'accepted_offer_type']]##Categorical variables# Encoding for non-numeric categorical variables
car_description2 = pd.get_dummies(car_description_reg, columns=['location_name', 'brand_name', 'accepted_offer_type'], drop_first=False)

# Ready data for multiple regression
X = car_description2.drop(['inventory_days'], axis=1)
y = np.log(car_description2[['inventory_days']].values.ravel())

# Split Train, Validation and Test Data
X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
X_train, X_vali, y_train, y_vali = train_test_split(X_rest, y_rest, test_size=0.20, random_state=0)

# Fit multiple linear regression to training data
#model_linear = sm.OLS(y_train, sm.add_constant(X_train))
model_linear = sm.OLS(y_train, X_train)
model_1 = model_linear.fit()
print(model_1.summary())


## plot histogram
plt.hist(model_1.resid, 
    density=True,     # the histogram integrates to 1 
                      # (so it can be compared to the normal distribution)
    bins=100,         #  draw a histogram with 100 bins of equal width
    label="residuals" # label for legend
    )
# now plot the normal distribution for comparison
xx = np.linspace(model_1.resid.min(), model_1.resid.max(), num=1000)
plt.plot(xx, scipy.stats.norm.pdf(xx, loc=0.0, scale=np.sqrt(model_1.scale)),
    label="normal distribution")
outliers = np.abs(model_1.resid)>4*np.sqrt(model_1.scale)
sns.rugplot(model_1.resid[outliers],
            color="C5", # otherwise the rugplot has the same color as the histogram
            label="outliers")
plt.legend(loc="upper right");

sm.qqplot(model_1.resid, line="s");

print("AIC",model_1.aic)
print("R2",model_1.rsquared)
print("RMSE", RMSE(np.exp(model_1.predict(X_test)) ,np.exp(y_test)))
print("MAPE ", MAPE(np.exp(model_1.predict(X_test)) ,np.exp(y_test)))

###Inventary Days Actual vs. Predicted##
fig, ax = plt.subplots(figsize=(15,10))
ax.scatter(np.exp(y_test), np.exp(model_1.predict(X_test)), edgecolors=(0, 0, 0))
ax.plot([np.exp(y_test).min(), np.exp(y_test).max()], [np.exp(y_test).min(), np.exp(y_test).max()], 
        'k--', lw=4,label='Ideal Model')
ax.set_xlabel('Actual Inventory Days')
ax.set_ylabel('Predicted Inventory Days')
ax.set_title('Inventary Days Actual vs. Predicted')
plt.legend(loc='upper left')
plt.show()

print("Running MODEL 2...")      
#############################
####### MODEL 2 #############
#############################

showroomlist =[' Florencia', 'Santa Fe', 'Fuera de Kavak', 'Fortuna', 'Lerma']
  
for i,showroom in enumerate(showroomlist):
    car_description_showroom = car_description.drop(car_description[car_description['location_name']!=showroom].index)
    
    #mean_val = car_description_showroom['inventory_days'].mean()
    #std_val = car_description_showroom['inventory_days'].std()
    #a=mean_val+2*std_val
    
    ###### Filter inventory days grater dan 70 #############
    car_description_showroom = car_description_showroom.drop(car_description_showroom[car_description_showroom['inventory_days']>=70].index)
    
    

    car_description_reg=car_description_showroom[['inventory_days','km', 'published_price' ##Numerical variables
                                                ]]##Categorical variables# Encoding for non-numeric categorical variables


    #car_description2 = pd.get_dummies(car_description_reg, columns=['accepted_offer_type','brand_name'], drop_first=False)
    car_description2=car_description_reg


    # Ready data for multiple regression
    X = car_description2.drop(['inventory_days'], axis=1)
    y = np.log(car_description2[['inventory_days']].values.ravel())

    # Split Train, Validation and Test Data
    X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    X_train, X_vali, y_train, y_vali = train_test_split(X_rest, y_rest, test_size=0.20, random_state=0)


    X_train.fillna(0)
    X_vali.fillna(0)
    X_test.fillna(0)


    ######### LINEAR MODEL ###############
    model_linear = sm.OLS(y_train, X_train)
    model_i = model_linear.fit()
    
    print("---------------------------------------------------")
    print("------------------",showroom,"---------------------")
    print("---------------------------------------------------")
    print("AIC",model_i.aic)
    print("R2",model_i.rsquared)
    print("RMSE", RMSE(np.exp(model_i.predict(X_test)) ,np.exp(y_test)))
    print("MAPE ", MAPE(np.exp(model_i.predict(X_test)) ,np.exp(y_test)))
    
    
    sm.qqplot(model_i.resid, line="s");
    
    fig, ax = plt.subplots(figsize=(5,3))
    ax.scatter(np.exp(y_test), np.exp(model_i.predict(X_test)), edgecolors=(0, 0, 0))
    ax.plot([np.exp(y_test).min(), np.exp(y_test).max()], [np.exp(y_test).min(), np.exp(y_test).max()], 
            'k--', lw=4,label='Ideal Model')
    ax.set_xlabel('Actual RUL')
    ax.set_ylabel('Predicted RUL')
    ax.set_title('Inventary Days Actual vs. Predicted')
    plt.legend(loc='upper left')
    plt.show()
       
    print(model_i.summary())
    
print("REGULARIZATION ...")
####################################
####### REGULARIZATION #############
####################################
    
    
    # Subset Data for Regularization model

car_description_model = car_description.drop(car_description[car_description['inventory_days']>=100].index)
    


car_description_reg=car_description_model[['inventory_days','km', 'published_price','tot_pageviews', ##Numerical variables
                                     'location_name', 'brand_name', 'accepted_offer_type','color']]##Categorical variables# Encoding for non-numeric categorical variables
car_description2 = pd.get_dummies(car_description_reg, columns=['location_name', 'brand_name', 'accepted_offer_type','color'], drop_first=False)

#car_description2 = pd.get_dummies(car_description_reg, columns=['location_name'], drop_first=False)
#car_description2=car_description_reg

# Ready data for multiple regression
X = car_description2.drop(['inventory_days'], axis=1)
y = np.log(car_description2[['inventory_days']].values.ravel())

# Split Train, Validation and Test Data
X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
X_train, X_vali, y_train, y_vali = train_test_split(X_rest, y_rest, test_size=0.20, random_state=0)

# Loop through different lambda values
lambdas = np.arange(0.1, 100.0, 0.1)

coefs = []
mse_train = []
mse_vali = []
for l in lambdas:
    ridge = linear_model.Lasso(alpha=l, fit_intercept=True)
    ridge.fit(X_train, y_train)
    mse_train.append(mean_squared_error(y_train, ridge.predict(X_train))) # train data
    mse_vali.append(mean_squared_error(y_vali, ridge.predict(X_vali))) # validation data
    
# Plot results
fig, ax1 = plt.subplots(1,3,figsize=(18,6))

ax1[0].plot(lambdas, mse_train, 'g-')
ax1[0].set_title('Train MSE curve')

ax1[1].plot(lambdas, mse_vali, 'b-')
ax1[1].set_title('Validation MSE curve')

#Place the two plots into one
ax1[2].plot(lambdas, mse_train, 'g-')
ax1[2].plot(lambdas, mse_vali, 'b-')
ax1[2].set_title('Train & Validation')


for ax in ax1.flat:
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('MSE')
    
    
plt.show()

# Fitting the model
model_l2 = linear_model.Lasso(alpha=13, fit_intercept=True) # higher alpha for stronger regularization
results_l2 = model_l2.fit(X_train, y_train)

sorted(zip(X_train.columns, results_l2.coef_), key=lambda x: x[1])
