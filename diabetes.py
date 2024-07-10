import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# function to encode string categorical var to numeric categorical var
def label_encode(colname, df):
    trancol = LabelEncoder().fit_transform(data[colname].values)
    colloct = list(df.columns).index(colname)
    df = df.drop(colname, axis = 'columns')
    df.insert(loc = colloct, column = colname, value = trancol)
    return df

# data loading and feature-outcome determination
data = pd.read_csv('diabetes_prediction_dataset.csv')
feature = data.drop(['diabetes'], axis = 'columns')
feature = label_encode('gender', feature)
feature = label_encode('smoking_history', feature)
outcome = data['diabetes'].values

# data description and dataset view
print(data.describe())
print(data.head())

# calculating feature importance in determining diabetes
imp_marker = RandomForestClassifier(random_state = 0, n_estimators = 100)
imp_marker.fit(feature.iloc[:,:].values, outcome)
importances = imp_marker.feature_importances_

# getting a descending sorted dataframe for feature & corresponding scores
feature_score = pd.DataFrame({
    'feature': feature.columns,
    'score': imp_marker.feature_importances_
})
feature_score = feature_score.sort_values(by = 'score', ascending = False)
print(feature_score)

# selecting features with importance score >= 0.05
selected_features = {'HbA1c_level', 'blood_glucose_level', 'bmi', 'age'}
feature = feature.drop((set(feature.columns) - selected_features), axis = 'columns').iloc[:,:].values

# spliting train-test data and training model
xtrain, xtest, ytrain, ytest = train_test_split(feature, outcome, test_size = 0.2)
classifier = RandomForestClassifier(random_state = 0, n_estimators = 100)
classifier.fit(xtrain, ytrain)
ypred = classifier.predict(xtest)

# checking confusion matrix
cf = confusion_matrix(ytest, ypred)
plt.title('Diabetes Prediction Results')
sns.heatmap(data = cf, cmap = 'Blues', fmt = 'd', annot = True)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

# calculating accuracy, error, precision, recall, f1-score
score = round((cf[0][0] + cf[1][1]) / (cf[0][0] + cf[0][1] + cf[1][0] + cf[1][1]), 2)
error = round((cf[0][1] + cf[1][0]) / (cf[0][0] + cf[0][1] + cf[1][0] + cf[1][1]), 2)
precision = round(cf[1][1] / (cf[1][0] + cf[1][1]), 2)
recall = round(cf[1][1] / (cf[0][1] + cf[1][1]), 2)
f1_score = round(2 * precision * recall / (precision + recall), 2)

print('score =', score)
print('error =', error)
print('precision =', precision)
print('recall =', recall)
print('f1 score =', f1_score)