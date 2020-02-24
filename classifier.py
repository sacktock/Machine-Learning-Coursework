import sys
import os
import sklearn
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.model_selection import StratifiedShuffleSplit

# Code Usage:
## 1. Ensure the following modules are installed on this version of python: sys, os, matplotlib, numpy, pandas, sklearn, scipy.stats
## 2. Ensure these files are in the same directory as this file: studentInfo.csv, studentRegistration.csv, studentAssessment.csv, studentVle.csv, courses.csv, assessments.csv, vle.csv
## 3. To run the this python script type: python3 classifier.py (linux) / python classifier.py (windows)
## -> This script is written in python 3.6.3 and requires no additional command line arguments
## -> This script creates 2 files in the cwd: student_data_hist.png and stratisfied_comparison_head.txt
## -> This script takes a few minutes to run completely, but prints out results as it goes along, so be patient
## -> note: the script used at different parts of the report may differ slightly to this one, the only
##          differences to the script used in parts of the report are the parameters to GridSearchCV and the initial feature selection.
##          It wll become clear in the report what parameters were changed and what features were dropped in the
##          'Parameter Search and Selection' part of the report.

def prepare_student_data(student_info, student_registration, student_assessment,
                          student_vle, courses, assessments, vle):
    # simplified student info
    student_df = student_info[['id_student', 'code_module', 'code_presentation', 'final_result']]

    # get the relevant assessment info
    # calculate the weighted coursework score (out of 100) for each student for each module
    assessment_df = pd.merge(left = student_assessment, right = assessments, left_on='id_assessment', right_on='id_assessment')[['id_student', 'code_module', 'code_presentation','assessment_type','score', 'weight']]
    assessment_df = assessment_df[assessment_df['assessment_type'] != 'Exam']
    
    # extract all the GGG assessments
    GGG_assessment = assessment_df[assessment_df['code_module'] == 'GGG']
    assessment_df = assessment_df[assessment_df['code_module'] != 'GGG']

    # calculate the estimated coursework score for GGG
    GGG_assessment['weighted_score'] = GGG_assessment['score']
    GGG_assessment = GGG_assessment[['id_student', 'code_module', 'code_presentation', 'weighted_score']]
    GGG_assessment = GGG_assessment.groupby(['id_student', 'code_module', 'code_presentation'])['weighted_score'].mean().reset_index()
    GGG_assessment['weighted_score'] = GGG_assessment['weighted_score'].round(0)
    
    # calculate the coursework score 
    assessment_df['weighted_score'] = assessment_df.score * assessment_df.weight 
    assessment_df['weighted_score'] = assessment_df['weighted_score'].div(100).round(0)
    assessment_df = assessment_df[['id_student', 'code_module', 'code_presentation', 'weighted_score']]
    assessment_df = assessment_df.groupby(['id_student', 'code_module', 'code_presentation'])['weighted_score'].sum().reset_index()

    # combine the GGG asessments
    assessment_df = pd.concat([assessment_df, GGG_assessment]).reset_index()
    
    # get the relevant vle interaction info
    # and calculate the total number of interactions with the vle material for each student for each module
    vle_df = pd.merge(left=student_vle, right=vle, left_on=['id_site', 'code_module', 'code_presentation'], right_on=['id_site', 'code_module', 'code_presentation'])[['id_student', 'code_module', 'code_presentation','sum_click']]
    vle_df = vle_df.groupby(['id_student', 'code_module', 'code_presentation'])['sum_click'].sum().reset_index()

    # use student_df for a more simplified dataframe
    # other wise use student_info
    # merge the student info with the data collected from the other tables
    student_stats = pd.merge(left=student_info, right=assessment_df, left_on=['id_student', 'code_module', 'code_presentation'], right_on=['id_student', 'code_module', 'code_presentation'])
    student_stats = pd.merge(left=student_stats, right=vle_df, left_on=['id_student', 'code_module', 'code_presentation'], right_on=['id_student', 'code_module', 'code_presentation'])

    # drop null rows
    student_stats = student_stats.dropna()
    return student_stats

# GET THE DATA
# open and read the csv files
try:
    student_info = pd.read_csv('./studentInfo.csv')
    student_registration = pd.read_csv('./studentRegistration.csv')
    student_assessment = pd.read_csv('./studentAssessment.csv')
    student_vle = pd.read_csv('./studentVle.csv')
    courses = pd.read_csv('./courses.csv')
    assessments = pd.read_csv('./assessments.csv')
    vle = pd.read_csv('./vle.csv')
except:
    print('error opening .csv files')
    sys.exit()
    
# PREPARE THE DATA FOR MACHINE LEARNING
# Collect the relevant student data using the database tables
student_data = prepare_student_data(student_info, student_registration, student_assessment,
                          student_vle, courses, assessments, vle)

# Drop the id_student code_module code_assessment tables
students = student_data[['weighted_score','sum_click', 'gender','highest_education','age_band', 'num_of_prev_attempts', 'studied_credits', 'disability', 'final_result']].copy()

# Encode the categorical data
encodings = {'gender' : {'M' : 0, 'F' : 1},
             'highest_education' : {'Post Graduate Qualification' : 4, 'HE Qualification' : 3,
                                    'A Level or Equivalent' : 2,'Lower Than A Level': 1,
                                    'No Formal quals' : 0},
             'disability' : {'Y': 0, 'N' : 1},
             'age_band' : {'0-35' : 0, '35-55' : 1, '55<=' : 2},
             'final_result' : {'Withdrawn' : 0, 'Fail': 1, 'Pass' : 2, 'Distinction': 3}
             }
students.replace(encodings, inplace=True)

# Describe the data
print('Prepared data:')
print(students.head())
print(students.describe())
print()

students.hist(bins=50, figsize=(20,15))
plt.savefig('student_data_hist')
plt.clf()

# Create the training set and test set
train_set, test_set = train_test_split(students, test_size=0.2, random_state=42)

# Stratify the test set and train set
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
strat_train_set = None
strat_test_set = None
for train_index, test_index in split.split(students, students['weighted_score']):
    strat_train_set = students.reindex(train_index)
    strat_test_set = students.reindex(test_index)
    
# Split the dataframe into features and label
strat_train_set = strat_train_set.dropna()
student_prepared = strat_train_set.drop('final_result',axis=1)
student_labels = strat_train_set['final_result'].copy()

# Compare the effectiveness of the stratisfied training set
def weighted_score_proportions(data):
    return data['weighted_score'].value_counts() / len(data)

print()
print('Stratisifed vs Random Training set (head):')
compare_props = pd.DataFrame({
    'Overall' : weighted_score_proportions(students),
    'Stratisfied' : weighted_score_proportions(strat_test_set),
    'Random' : weighted_score_proportions(test_set)
    }).sort_index()
compare_props['Rand. %error'] = 100 * compare_props['Random'] / compare_props['Overall'] -100
compare_props['Strat. %error'] = 100 * compare_props['Stratisfied'] / compare_props['Overall'] -100

print(compare_props.head())
print()
file = open('stratisfied_comparison_head.txt', 'w+')
file.write(compare_props.head().to_latex(index=False))
file.close()

# Select a suitable model // Decision tree
tree_clf = DecisionTreeClassifier(random_state=42)

# Train the model
tree_clf.fit(student_prepared, student_labels)

some_data = student_prepared[:10]
some_labels = student_labels[:10]
print('Some example predictions:')
print('Predictions: ', tree_clf.predict(some_data))
print('Labels: ', list(some_labels))
print('Examples: ',some_data)

# Test the model
student_predictions = tree_clf.predict(student_prepared)
tree_mse = mean_squared_error(student_labels, student_predictions)
tree_rmse = np.sqrt(tree_mse)
print()
print('tree_rmse: ',tree_rmse)

# Fine tune the model
tree_scores = cross_val_score(tree_clf, student_prepared, student_labels,
                         scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-tree_scores)

def display_scores(scores):
    print()
    print('neg_mean_squared_error scores:')
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    print()

display_scores(tree_rmse_scores)

# Select another suitable model // Random forest
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
forest_clf.fit(student_prepared, student_labels.ravel())

some_data = student_prepared[:10]
some_labels = student_labels[:10]

print('Some example predictions:')
print('Predictions: ', forest_clf.predict(some_data))
print('Labels: ', list(some_labels))
print('Examples: ',some_data)

# Test the model
student_predictions = forest_clf.predict(student_prepared)
forest_mse = mean_squared_error(student_labels.ravel(), student_predictions)
forest_rmse = np.sqrt(forest_mse)
print()
print('forest_rmse: ',forest_rmse)

# Fine tune the model
forest_scores = cross_val_score(forest_clf, student_prepared, student_labels.ravel(),
                         scoring="neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)

display_scores(forest_rmse_scores)

# Use the grid search CV to fine tune the model for the random forest
# note: in the report we try a randomized search first with,
#       param_distribs = {
#           'n_estimators': randint(low=1, high=200),
#           'max_features': randint(low=1,high=8)
#           }.
#     -> Then we perform a grid search with,
#       param_grid = [
#           {'n_estimators' : [100, 120, 160, 180, 200], 'max_features': [4, 5, 6]},
#           {'bootstrap' : [False], 'n_estimators' : [100, 120, 160, 180, 200], 'max_features' : [4, 5, 6]}
#           ].
# The following code is simplfied with smaller parameters for submission in the interest of running time

param_grid = [
    {'n_estimators' : [3,10,30], 'max_features': [2,4,5,6]},
    {'bootstrap' : [False], 'n_estimators' : [3,10], 'max_features' : [2,3,4]}
    ]
forest_clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(forest_clf, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(student_prepared, student_labels)

print()
print('Best params: ',grid_search.best_params_)
print('Best estimator: ',grid_search.best_estimator_)
print('Grid search results: ',pd.DataFrame(grid_search.cv_results_))
print()

# Now try a randomised search

param_distribs = {
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=1,high=8)
    }
forest_clf = RandomForestClassifier(random_state=42)
rnd_search = RandomizedSearchCV(forest_clf, param_distributions=param_distribs,
                                    n_iter=10, cv=5, scoring='neg_mean_squared_error',
                                    random_state=42)
rnd_search.fit(student_prepared, student_labels)

print()
print('Best params: ',rnd_search.best_params_)
print('Best estimator: ', rnd_search.best_estimator_)
print('Random search results: ', pd.DataFrame(rnd_search.cv_results_))
print()

# Pick the best model
print()
print('Feature importances (grid search): ',grid_search.best_estimator_.feature_importances_)
print('Feature importances (rnd search): ',rnd_search.best_estimator_.feature_importances_)
print()
final_model = rnd_search.best_estimator_

# Finally test the model on the test set
strat_test_set = strat_test_set.dropna()
X_test = strat_test_set.drop('final_result', axis=1)
y_test = strat_test_set['final_result'].copy()

final_predictions = final_model.predict(X_test)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print('final model strat_test_rmse: ',final_rmse)

# Compare the optimized model to the decision tree model
final_predictions = tree_clf.predict(X_test)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print('tree clf strat_test_rmse: ',final_rmse)

