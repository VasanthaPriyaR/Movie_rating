# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 21:39:44 2018
  
@author: vasanthapriya Raja
"""

import numpy as np
import pandas as pd
# Importing the dataset
df = pd.read_csv('movie_metadata.csv')
# Drop null values from 'director_name' column, since imputing director's name
# seems illogical
df = df.dropna(axis=0, subset=['director_name'])
Numerical_data=df.select_dtypes([np.number])
 # Creating the Target variable
y = df.iloc[:,-3].values
             
        # Taking care of missing data and Feature scaling
# Numerical variables 
        #Segregating numeric features that needs feature scaling
numeric_features=df._get_numeric_data().columns.values.tolist()

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(Numerical_data)
Numerical_data[numeric_features] = imputer.transform(Numerical_data[numeric_features])
from sklearn.preprocessing import StandardScaler
scl=StandardScaler()
Numerical_data[numeric_features]=scl.fit_transform(Numerical_data[numeric_features])
#Removing unimportant and target variables from numeric_features's list
Numerical= Numerical_data.drop(['facenumber_in_poster','imdb_score','title_year'],axis=1)
numeric_features.remove("facenumber_in_poster")
numeric_features.remove("imdb_score")
numeric_features.remove("title_year")

# Categorical variables
  # Creating list of Categorical_features
text_features=df.columns.values.tolist()
text_features=[i for i in text_features if i not in numeric_features]
string_features=["movie_title", "plot_keywords"]
Categorical_dataset=df.drop(['num_critic_for_reviews','duration','director_facebook_likes',
                             'actor_3_facebook_likes','actor_1_facebook_likes',
                             'actor_2_facebook_likes','gross','num_voted_users',
                             'cast_total_facebook_likes','plot_keywords','movie_imdb_link',
                             'language','content_rating','budget',
                             'imdb_score','aspect_ratio','movie_facebook_likes',
                            'facenumber_in_poster','num_user_for_reviews'],axis=1) 


categorical_features=[i for i in text_features if i not in string_features]
categorical_features.append("title_year")
categorical_features.append("movie_title")
categorical_features.remove("content_rating")
categorical_features.remove("movie_imdb_link")
categorical_features.remove("language")
categorical_features.remove("title_year")
categorical_features.remove("imdb_score")
categorical_features.remove("facenumber_in_poster")

#Creating Custom imputer class to impute Categorical_features
from sklearn.preprocessing import Imputer
from sklearn.base import TransformerMixin

class CustomImputer(TransformerMixin):
         def __init__(self, cols=None, strategy='mean'):
               self.cols = cols
               self.strategy = strategy

         def transform(self, df):
               X = df.copy()
               impute = Imputer(strategy=self.strategy)
               if self.cols == None:
                      self.cols = list(X.columns)
               for col in self.cols:
                      if X[col].dtype == np.dtype('O') : 
                             X[col].fillna(X[col].value_counts().index[0], inplace=True)
                      else : X[col] = impute.fit_transform(X[[col]])

               return X

         def fit(self, *_):
               return self
cci = CustomImputer(cols=categorical_features,strategy='most_frequent') # here default strategy = mean
Categorical_dataset=cci.fit_transform(Categorical_dataset)
 # Encoding categorical data-creating dummy variables
"""from sklearn.preprocessing import LabelEncoder,OneHotEncoder 
labelencoder_1=LabelEncoder()
labelencoder_2=LabelEncoder()
labelencoder_3=LabelEncoder()
labelencoder_4=LabelEncoder()
labelencoder_5=LabelEncoder()
labelencoder_6=LabelEncoder()
labelencoder_7=LabelEncoder()
labelencoder_8=LabelEncoder()
Categorical_dataset["color"]=labelencoder_1.fit_transform(Categorical_dataset["color"])
Categorical_dataset["director_name"]=labelencoder_2.fit_transform(Categorical_dataset["director_name"])
Categorical_dataset["actor_2_name"]=labelencoder_3.fit_transform(Categorical_dataset["actor_2_name"])
Categorical_dataset["genres"]=labelencoder_4.fit_transform(Categorical_dataset["genres"])
Categorical_dataset["actor_1_name"]=labelencoder_5.fit_transform(Categorical_dataset["actor_1_name"])
Categorical_dataset["actor_3_name"]=labelencoder_1.fit_transform(Categorical_dataset["actor_3_name"])
Categorical_dataset["title_year"]=labelencoder_1.fit_transform(Categorical_dataset["title_year"])
Categorical_dataset["movie_title"]=labelencoder_1.fit_transform(Categorical_dataset["movie_title"])

onehotencoder=OneHotEncoder(categorical_features = [0])
Categorical_dataset_coded = onehotencoder.fit_transform(Categorical_dataset)"""

for feat in categorical_features:
    Categorical_dataset=pd.concat([Categorical_dataset, pd.get_dummies(Categorical_dataset[feat], prefix=feat, dummy_na=True)],axis=1)
cat_dummy_list=[i for i in Categorical_dataset.columns.values.tolist() if i not in numeric_features]
cat_dummy_list=[i for i in cat_dummy_list if i not in text_features]
cat_dummy_list[-5:]
#Correlation between categorical variables
import operator

from scipy.stats import pearsonr
correl={}
for f in cat_dummy_list:
    correl[f]=pearsonr(Categorical_dataset[f],y)
sorted_cor_c = sorted(correl.items(), key=operator.itemgetter(1), reverse=True)

print (sorted_cor_c[0:10])
print (sorted_cor_c[-10:])
print (sorted_cor_c)
#Instead of viewing correl matrix of categorical variables taking imp variables from sorted ones and assuming them as predictors.
Categorical_features_new=["color_ Black and White", "genres_Drama",
            "genres_Crime|Drama", "genres_Drama|Romance", "title_year_2015.0", 
           "director_name_Jason Friedberg","genres_Horror","genres_Comedy|Romance",
           "director_name_Uwe Boll" ,"color_Color","country_USA"]
#Distribution of Target variable
import matplotlib.pyplot as plt
font = {'fontname':'Arial', 'size':'14'}
title_font = { 'weight' : 'bold','size':'16'}
plt.hist(y, bins=20)
plt.title("Distribution of the IMDB ratings")
plt.show()
# Correlation of numerical variables
import operator

from scipy.stats import pearsonr
correl={}
for f in numeric_features:
    correl[f]=pearsonr(Numerical[f], y)
sorted_cor = sorted(correl.items(), key=operator.itemgetter(1), reverse=True)
print (sorted_cor)
import seaborn as sns
import matplotlib.pyplot as plt
def corrmap(features, title):
    sns.set(context="paper", font="monospace")
    corrmat = Numerical[features].corr()
    f, ax = plt.subplots(figsize=(12, 9))
    plt.title(title,**title_font)
# Draw the heatmap using seaborn
    sns.heatmap(corrmat, vmax=.8, square=True)
corrmap(numeric_features,"Correlation matrix for numeric features")
# Removing features based on the correlation matrix 
#Looking at the matrix, the num_critic for movies matches more than 50% with 5 other variables 
#since all the variables are scaled I am taking the average of all to be a new variable for the model
Numerical["avg_rev_gross"]=(Numerical['num_critic_for_reviews']+Numerical["num_voted_users"]
                +Numerical["num_user_for_reviews"]+Numerical["gross"]+Numerical["movie_facebook_likes"])/5
#similarly the actors facebook likes seems highly correlated and so averaging them too into a single variable
Numerical["other_actors_facebook_likes"]=(Numerical["actor_2_facebook_likes"]+
         Numerical["actor_3_facebook_likes"]+Numerical["cast_total_facebook_likes"])/3
# Creating new numerical features list with updated variables and removing correlated ones
num_features_new=[x for x in numeric_features if x not in ["cast_total_facebook_likes",
                                                         'num_critic_for_reviews',
                                                         "num_voted_users",
                                                         "num_user_for_reviews",
                                                        "gross","movie_facebook_likes",
                                                        "actor_2_facebook_likes",
                                                        "actor_3_facebook_likes"]]
num_features_new.extend(["avg_rev_gross", "other_actors_facebook_likes"])
# Checking Correlation between new list of continuous variables 
corrmap(num_features_new, "Correlation matrix with new numeric features")
# Choosing predictors of numerical variables- 
#actor 1 and other actor facebook likes are correlated, so we can chose only one as predictor

 #Creating the final data set
#y=pd.DataFrame(y)
X= pd.concat([Categorical_dataset[Categorical_features_new],Numerical[num_features_new]],axis=1)
X1= pd.concat([Categorical_dataset,Numerical[num_features_new]],axis=1)
Predictors= ["duration","director_facebook_likes","actor_1_facebook_likes",
             "budget","aspect_ratio","avg_rev_gross","color_ Black and White", "genres_Drama",
            "genres_Crime|Drama", "genres_Drama|Romance", "title_year_2015.0", 
           "director_name_Jason Friedberg","genres_Horror","genres_Comedy|Romance",
           "director_name_Uwe Boll" ,"color_Color","country_USA","other_actors_facebook_likes"]
#Linear Model
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
reg=regressor.fit(X_train, y_train)
# Predicting the Test set results
y_pred = regressor.predict(X_test)
#Creating dataframe with predicted and actual y-values
y_assess = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
y_assess  
#Evaluating model metrics
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 
print('RSquared:',metrics.r2_score(y_test, y_pred))
# Fitting Multiple Linear Regression to the Training set 2
#Removing budet and checking again.
Predictors_new= ["duration","director_facebook_likes","actor_1_facebook_likes",
             "aspect_ratio","avg_rev_gross","color_ Black and White", "genres_Drama",
            "genres_Crime|Drama", "genres_Drama|Romance", "title_year_2015.0", 
           "director_name_Jason Friedberg","genres_Horror","genres_Comedy|Romance",
           "director_name_Uwe Boll" ,"color_Color","country_USA","other_actors_facebook_likes"]
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
reg2=regressor.fit(X_train[Predictors_new], y_train)
# Predicting the Test set results
y_pred2 = regressor.predict(X_test[Predictors_new])
#Creating dataframe with predicted and actual y-values
y_assess = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred2})  
y_assess  
#Evaluating model metrics
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred2))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred2))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2))) 
print('RSquared:',metrics.r2_score(y_test, y_pred2))
#Model summary
import statsmodels.api as sm
# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)
# Print out the statistics
model.summary()
# The budget  variable seems to have lesser significane compared to other variables
#Removing budet and checking again.
Predictors_new= ["duration","director_facebook_likes","actor_1_facebook_likes",
             "aspect_ratio","avg_rev_gross","color_ Black and White", "genres_Drama",
            "genres_Crime|Drama", "genres_Drama|Romance", "title_year_2015.0", 
           "director_name_Jason Friedberg","genres_Horror","genres_Comedy|Romance",
           "director_name_Uwe Boll" ,"color_Color","country_USA","other_actors_facebook_likes"]
#Model summary
import statsmodels.api as sm
# Note the difference in argument order
model = sm.OLS(y, X[Predictors_new]).fit() ## sm.OLS(output, input)
predictions = model.predict(X[Predictors_new])
# Print out the statistics
model.summary()
#Random Forest Model
from sklearn.ensemble import RandomForestRegressor
RFR=RandomForestRegressor(max_features="sqrt")
parameters={ "max_depth":[5,8,25], 
             "min_samples_split":[2,5], "n_estimators":[800,1200]}
from sklearn.grid_search import GridSearchCV
clf = GridSearchCV(RFR, parameters)
clf.fit(X, y)
from operator import itemgetter
# Utility function to report best scores
def report_sum(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
report_sum(clf.grid_scores_)
''' CONCLUSION:
    Upon comparing both model metrics(Rsquare-validation), the Random forest model 
seems to perform better.But, the score itself is very low which is 0.32 on average which is 
not a better value for variance.since the value is very low where the range is (0,1) and 
we are getting only 0.32, which fits the model with higher residual.
We can improve the model much better with added value variables. Also the list of predictors 
here seems logical and worked well with model'''
 