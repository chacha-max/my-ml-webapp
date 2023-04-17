import pandas as pd
import numpy as np

data = pd.read_csv('dataset_admissions.csv')

dummy_rank = pd.get_dummies(data['rank'],prefix="rank") # converte a variável categórica (1,2,3,4) para valores binários

columns_to_keep = ['admit','gre','gpa']
data = data[columns_to_keep].join(dummy_rank[['rank_1','rank_2','rank_3','rank_4']])

# Creating variables to store the results
majority = data[data['admit']==0]
minority = data[data['admit']==1]

from sklearn.utils import resample

# Applying a resampling strategy (Oversampling) to obtain a more balanced data
minority_upsample = resample(minority, replace = True, n_samples=273, random_state=123)
new_data = pd.concat([majority, minority_upsample])

# Creating X
X = new_data.drop("admit",axis=1)
print(X.columns)
# Creating Y
Y = new_data["admit"]

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_real = train_test_split(X,Y,test_size=0.2) # test size will be 20% and train size will be 80%


from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

model_random_forest = RandomForestClassifier().fit(X_train,Y_train)
y_pred_random_forest = model_random_forest.predict(X_test)

acc_rf = metrics.accuracy_score(Y_real,y_pred_random_forest)
print(acc_rf)

#retrain the model using the whole dataset
X = X.values
Y = Y.values

model_random_forest = RandomForestClassifier().fit(X,Y)

# pickling the model
import pickle
pickle_out = open("classifier.pkl", "wb")
pickle.dump(model_random_forest, pickle_out)
pickle_out.close()