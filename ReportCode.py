
# coding: utf-8

# In[259]:

import pandas as pd
import sklearn
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from scipy.stats import chisquare, pearsonr, spearmanr
import re

import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')


# question 1 - present background
# question 2 - 3 preferences of higher education (type, institute, level, major 
# question 3 - how sure are you about the options you listed above 
# question 4 - annual costs for studying at these institutions 
# question 5 - monthly salary expectation 
# question 6 - score with which you shall apply 

# In[83]:

'''We load our data into a pandas dataframe. The labeling of the columns were a little hard for me to follow since 
I dont speak Spanish, so I decided to rename then which will help me follow a lot 
easier. There were specific ones that needed to be replaced, so, well, I brute forced it.'''

survey = pd.read_csv('edu_chile_survey_output.csv', header = 0, low_memory = False)
survey.rename(columns={'rut_orig':'ID', 'q1':'q1_postsec', 'q2_tipo_1_orig':'q2_institute_type_1', 
                       'q2_tipo_2_orig':'q2_institute_type_2', 'q2_tipo_3_orig':'q2_institute_type_3', 
                       'q2_nivel_1':'q2_level_1', 'q2_nivel_2':'q2_level_2', 'q2_nivel_3':'q2_level_3', 
                       'q2_carerra_1':'q2_career_1', 'q2_carerra_2':'q2_career_2', 'q2_carerra_3':'q2_career_3', 
                       'q4_nose_1_orig':'q4_IDK_1', 'q4_nose_2_orig':'q4_IDK_2', 'q4_nose_3_orig':'q4_IDK_3', 
                       'q4_cost_1_orig':'q4_TuitionEstimate_1', 'q4_cost_2_orig':'q4_TuitionEstimate_2', 
                       'q4_cost_3_orig':'q4_TuitionEstimate_3',  'q5_mi_ing_1_orig':'q5_ownwages_1', 
                       'q5_mi_ing_2_orig':'q5_ownwages_2', 'q5_mi_ing_3_orig':'q5_ownwages_3', 
                       'q5_tip_ing_1_orig':'q5_typwages_1', 'q5_tip_ing_2_orig':'q5_typwages_2', 
                       'q5_tip_ing_3_orig':'q5_typwages_3', 'q6_math_orig':'q6_math', 
                       'q5_tip_nose_1_orig': 'q5_IDK_own_1', 'q5_tip_nose_2_orig': 'q5_IDK_own_2', 
                       'q5_tip_nose_3_orig': 'q5_IDK_own_3', 'q5_mi_nose_1_orig': 'q5_IDK_typ_1',
                       'q5_mi_nose_2_orig': 'q5_IDK_typ_2', 'q5_mi_nose_3_orig': 'q5_IDK_typ_3',
                       'q6_lang_orig': 'q6_lang','PSU_leng_2013': 'lang_score', 'PSU_mate_2013': 'math_score', 
                       'PSU_2013':'comp_score','SIMCEMath10':'math_10', 'SIMCELang10':'lang_10', 
                       'mom_educ_simce':'educ_mom', 'dad_educ_simce':'educ_dad'
                      }, inplace=True)

(survey.columns.values)


# In[84]:

pd.set_option('display.max_columns', None)
survey[]


# # DETERMINING THOSE WHO SEARCHED THE DATABASE 

# In[94]:

'''Most important step in the program, we mark the people who searched the database and those who did not search 
the database. '''

reports = (survey[['search1_psu_math','search1_psu_lang','search1_area','search1_nivel','search1_carrer']]).dropna()
rows = list(reports.index)
len(rows)
dim = survey.shape
r = dim[0]
results = [0] * r 

for row in rows: 
    results[row] = 1

survey['survey_participant'] = results


# # DATA CLEANING 

# In[ ]:




# In[95]:

pd.set_option('display.max_columns', None)

'''let's initiate the data cleaning process. First, we check if there are any rows which contain only NaN values, and 
get rid of them if that is the case. 
'''
survey = survey.dropna(axis = 0, how = 'all')


# In[96]:

#survey[['math_10', 'lang_10']] = survey[['math_10', 'lang_10']] + 10


# In[ ]:




# The next step helps us clear and categorize some of our data. As is noticeable, there are numerical and categorical variables in our data set that need to be separated in the interest of creating and training our model. After separating these variables, we set out to clean our numerical data. Some entries such as '$4.000.000' are corrected to '4000000' by the use of regular expressions. In the next section, we divide our data into inputs and outputs (outputs implying students using the database) and then into categorical and numerical. 

# In[336]:

numerical_inputs = ['q4_TuitionEstimate_1', 'q4_TuitionEstimate_2', 'q4_TuitionEstimate_3', 
                    'q5_ownwages_1', 'q5_ownwages_2', 'q5_ownwages_3',  'q5_typwages_1', 
                    'q5_typwages_2', 'q5_typwages_3', 'q6_math', 'q6_lang','lang_score', 
                    'math_score', 'comp_score', 'math_10', 'lang_10']

# , 'search1_psu_math', 'search1_psu_lang'
categorical_inputs = ['q1_postsec','q2_institute_type_1', 'q2_institute_type_2', 'q2_institute_type_3',
                      'q2_level_1', 'q2_level_2', 'q2_level_3', 'q2_career_1', 'q2_career_2', 'q2_career_3',
                      'q2_inst_1', 'q2_inst_2', 'q2_inst_3', 'q3', 'q4_IDK_1', 'q4_IDK_2', 'q4_IDK_3', 
                      'q5_IDK_own_1', 'q5_IDK_own_2', 'q5_IDK_own_3', 'q5_IDK_typ_1', 'q5_IDK_typ_2', 'q5_IDK_typ_3', 
                      'educ_mom', 'educ_dad', 'schl_type', 'rbdRating']

numerical_outputs = [ 'search1_psu_math', 'search1_psu_lang', 'search2_psu_math', 'search2_psu_lang', 
                      'search3_psu_math', 'search3_psu_lang', 'search4_psu_math', 'search4_psu_lang', 
                      'search5_psu_math', 'search5_psu_lang', 'search6_psu_math', 'search6_psu_lang', 
                      'search7_psu_math', 'search7_psu_lang', 'search8_psu_math', 'search8_psu_lang', 
                      'search9_psu_math', 'search9_psu_lang', 'search10_psu_math', 'search10_psu_lang']

categorical_outputs = ['search1_nivel', 'search1_carrer', 'search2_nivel', 'search2_carrer',
                       'search3_nivel', 'search3_carrer', 'search4_nivel', 'search4_carrer',
                       'search5_nivel', 'search5_carrer', 'search6_nivel', 'search6_carrer',
                       'search7_nivel', 'search7_carrer', 'search8_nivel', 'search8_carrer',
                       'search9_nivel', 'search9_carrer', 'search10_nivel', 'search10_carrer']


# In[182]:

# In this small section, we clean our numerical input and output data. It takes a little while to run.

def clean_inputs(sub):
    cleaned_data = pd.DataFrame(sub)
    for column in sub: 
        l = []
        x = sub[column]
        for value in x:
            q = re.findall(r'\d+',str(value)) # we extract just the numbers using regular expression 
            if q ==[]: 
                l.extend([None])
                continue
            l.append(int(q[0]))
        cleaned_data[column] = l
    return cleaned_data

# we fix all the issues with our numerical inputs here 
survey[numerical_inputs]  = clean_inputs(survey[numerical_inputs])
survey[numerical_outputs]  = clean_inputs(survey[numerical_outputs])


# In[ ]:




# In[183]:

'''In this section, we focus our attention towards our categorical data. first, let's find out the fraction of inputs 
 in the level, careers, and institute that is missing''' 

print 'Institute 1: ', "{0:.2f}".format(sum((survey['q2_inst_1']).isnull())/
    float(len(survey['q2_career_1']))*100),'% and Level 1: ', "{0:.2f}".format(sum((survey['q2_level_1']).isnull())/
    float(len(survey['q2_career_1']))*100),'% and Career 1: ', "{0:.2f}".format(sum((survey['q2_career_1']).isnull())/
    float(len(survey['q2_career_1']))*100),'%'
print 'Institute 2: ', "{0:.2f}".format(sum((survey['q2_inst_2']).isnull())/
    float(len(survey['q2_career_1']))*100),'% and Level 2: ', "{0:.2f}".format(sum((survey['q2_level_2']).isnull())/
    float(len(survey['q2_career_1']))*100),'% and Career 2: ', "{0:.2f}".format(sum((survey['q2_career_2']).isnull())/
    float(len(survey['q2_career_1']))*100),'%'
print 'Institute 3: ', "{0:.2f}".format(sum((survey['q2_inst_3']).isnull())/
    float(len(survey['q2_career_1']))*100),'% and Level 3: ', "{0:.2f}".format(sum((survey['q2_level_3']).isnull())/
    float(len(survey['q2_career_1']))*100),'% and Career 3: ', "{0:.2f}".format(sum((survey['q2_career_3']).isnull())/
    float(len(survey['q2_career_1']))*100),'%'


# In[353]:

'''We make an assumption here, that we shall later test. We noticed that about 23% of students dont have entries for 
their first choice of institute, level and career, 30% for their second, and 34% for their third, we make a 
transformation on our data. We convert those values to unknown, which in itself represents a (very valid) category 
of students who are not sure of their plans. '''

transform_categories = ['q1_postsec', 'q2_institute_type_1', 'q2_institute_type_2', 'q2_institute_type_3',
                        'q2_level_1', 'q2_level_2', 'q2_level_3', 'q2_career_1', 'q2_career_2', 
                        'q2_career_3', 'q2_inst_1', 'q2_inst_2', 'q2_inst_3', 'q3']

for column in survey[transform_categories]: 
    rows = survey[column].isnull()
    row = [i for i, x in enumerate(rows) if x]
    survey[column].loc[rows] = 'unknown'


# In[379]:




# In[ ]:




# # DIMENSIONATILY REDUCTION 

# In[ ]:

'''After we generate slightly cleaner data, it's time to do some dimensionality reduction. Instead of using the large 
number of features provided, we condense to the relative ones, to help generate more accurate trees less prone to 
overfitting. To test this, we run a spearman correlation to see the correlated terms. We notice a correlation between 
the institution and type of institute, as well as level and career. I considered running a Multiple Correpondence 
Analysis (PCA for categorical data), but their were issues with the only support for that in python. Instead, while 
training the forest, I decided to test the dimensions I reduced to determine the best fit for the model. '''

categories = ['q2_institute_type_1', 'q2_institute_type_2', 'q2_institute_type_3',
                       'q2_career_1', 'q2_career_2', 'q2_career_3',
                      'q2_inst_1', 'q2_inst_2', 'q2_inst_3']

#'q1_postsec','q2_level_1', 'q2_level_2', 'q2_level_3',
observed_categories = []
for x in survey[categories]: 
    observed_categories.extend([x])
    col_1 = survey[x]
    for y in survey[categories]: 
        if x == y or y in observed_categories: 
            continue
        col_2 = survey[y]
        #print "The correlation between", x , 'and', y , 'is', spearmanr(col_1,col_2)


# In[ ]:

'''On studying the correlation values generated, we notice that while in some cases there seem to be good correlation, 
the numbers are not extreme enough to justify reducing the data to fewer principal features. In our next section, we 
move onto variance analysis, where we study the variations between the two groups (of people who did and didn't use 
the database). '''


# # VARIANCE ANALYSIS 

# Given this is a classfication problem, one good way of solving this would be to segregate the two groups by observing 
# the variations in their data. For example, if test scores are significantly higher for those who use the database than those who don't (not the case here), then test scores would be a good category to include in our model. We look at our different inputs in the section below (post secondary details, institute, career and level preferences, wages and cost estimates, test scores, and 

# In[384]:

'''We break our data up into the yes and the no group. This helps us investigate how the behaviour of these two 
different groups are related. '''

yes_group = survey.ix[survey['survey_participant'] == 1,]
no_group = survey.ix[survey['survey_participant'] == 0,]

yes_rows =  yes_group.shape[0]
no_rows = no_group.shape[0]


# In[410]:

from collections import Counter

def normalize_ratios(a, b): 
    for key in a: 
        if key not in b: 
            continue
        value1 = a[key]
        value2 = b[key]
        a[key] = round(value1/float(yes_rows),3)
        b[key] = round(value2/float(no_rows),3)
    return a, b 

IDK_groups = ['q4_IDK_1', 'q4_IDK_2', 'q4_IDK_3', 'q5_IDK_own_1', 'q5_IDK_own_2', 
              'q5_IDK_own_3','q5_IDK_typ_1', 'q5_IDK_typ_2', 'q5_IDK_typ_3']

for column in categorical_inputs: 
    if column in ['q1_postsec','q2_career_1', 'q2_career_2', 'q2_career_3', 'q2_inst_1', 'q2_inst_2', 'q2_inst_3']: 
        continue
    y_IDK = yes_group[column]
    n_IDK = no_group[column]    
    y_IDK = dict(Counter(y_IDK))
    n_IDK = dict(Counter(y_IDK))
    y_IDK, n_IDK = normalize_ratios(y_IDK, n_IDK)
    print y_IDK, n_IDK


# In[398]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[358]:

# test scores analysis 
test_cols = ['q6_math', 'q6_lang','lang_score', 'math_score', 'comp_score', 'math_10', 'lang_10']
yes_avg_scores = yes_group[test_cols].mean(axis = 0)
no_avg_scores = no_group[test_cols].mean(axis = 0)

yes_std_scores = yes_group[test_cols].std(axis = 0)
no_std_scores = no_group[test_cols].std(axis = 0)


print "Students who said Yes:", yes_avg_scores, yes_std_scores
print "Students who said No:", no_avg_scores, no_std_scores
#fig, ax = plt.subplots()
#plt.hist(yes_avg_scores)


# In[359]:

# cost and wage analysis 
costs_wages = ['q4_TuitionEstimate_1', 'q4_TuitionEstimate_2', 'q4_TuitionEstimate_3', 
                    'q5_ownwages_1', 'q5_ownwages_2', 'q5_ownwages_3',  'q5_typwages_1', 
                    'q5_typwages_2', 'q5_typwages_3']

'''Too much overlap to use these variables to tell the data apart.'''

for cw in costs_wages:  
    yes_avg_wages = yes_group[cw].mean(axis = 0)
    no_avg_wages = no_group[cw].mean(axis = 0)
    yes_std_wages = (yes_group[cw].std(axis = 0))
    no_std_wages = no_group[cw].std(axis = 0)
    print "YES GROUP AVG:", cw, "{0:.2f}".format(yes_avg_wages), "YES GROUP SD:", cw, "{0:.2f}".format(yes_std_wages)
    print "NO GROUP AVG:", cw, "{0:.2f}".format(no_avg_wages), "NO GROUP SD:", cw, "{0:.2f}".format(no_std_wages)


# In[ ]:




# In[ ]:




# In[345]:

'''After we generate slightly cleaner data, it's time to do some dimensionality reduction. Instead of using the large 
number of features provided, we condense to the relative ones, to help generate more accurate trees less prone to 
overfitting. To test this, we run a spearman correlation to see the correlated terms. We notice a correlation between 
the institution and type of institute, as well as level and career. I considered running a Multiple Correpondence 
Analysis (PCA for categorical data), but their were issues with the only support for that in python. Instead, while 
training the forest, I decided to test the dimensions I reduced to determine the best fit for the model. '''

categories = ['q2_institute_type_1', 'q2_institute_type_2', 'q2_institute_type_3',
                       'q2_career_1', 'q2_career_2', 'q2_career_3',
                      'q2_inst_1', 'q2_inst_2', 'q2_inst_3']

#'q1_postsec','q2_level_1', 'q2_level_2', 'q2_level_3',
observed_categories = []
for x in survey[categories]: 
    observed_categories.extend([x])
    col_1 = survey[x]
    for y in survey[categories]: 
        if x == y or y in observed_categories: 
            continue
        col_2 = survey[y]
        #print "The correlation between", x , 'and', y , 'is', spearmanr(col_1,col_2)


# In[361]:

from sklearn.decomposition import PCA

'''Then, we consider inputs that have test scores. Unlike categorical inputs (discussed later), we drop the students 
who input no form of score. It makes it harder to come up with numerical estimates of what the data maybe, and often 
we land up having the data tell us what we want to hear, instead of what its truly trying to say. To minimize the 
number of students that are dropped, we conduct a correlation study between the expected, actual and 10th grade test 
scores. The highly correlated variables can be condensed to one through an understanding of dimensionality reduction. 
'''
math_test_scores = ['q6_math','math_score', 'math_10' ] # ,'math_10', 'lang_10'
lang_test_scores = ['q6_lang','lang_score', 'lang_10']
math_scores = survey[math_test_scores].dropna(axis = 0, how = 'any')
lang_scores = survey[lang_test_scores].dropna(axis = 0, how = 'any')

x = (survey[numerical_inputs].dropna())
final_rows = list(x.index)
x.index = range(0,len(final_rows))


'''We condense the size of the scores, to reduce the number of dimensions in final model by running a PCA. Selecting 
2 components explains about 87% of the variation, and 3 components explain about 97% of the variations. We test 
each of these in our final model and chose the best fit.'''
pca = PCA(n_components = 3)
X = pca.fit(x)
X = pd.DataFrame(pca.transform(x))


# In[362]:

'''Now we turn our attention towards implementing and training our model. With the data that we have, there are 
two approaches that we can take - Random Forests or the Naive Bayes Classifier. Since the final implementation after 
curing the data isn't too complicated, we test out both to see which one works better! But before we implement then, 
there's that crucial step of breaking the data in training and testing sets to test our final models.
'''

from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score


final_survey = survey.ix[final_rows,]
default_rows = range(0,len(final_survey))
final_survey.index = default_rows

# 'search1_area', 'search1_nivel', 'search1_carrer'
features = ['q1_postsec', 'q2_institute_type_1', 'q2_level_1', 'q2_career_1', 'q2_institute_type_2', 'q2_level_2','q2_career_2',
           'q2_institute_type_3', 'q2_level_3','q2_career_3', 'q3']


#final_survey['isTrain'] = np.random.uniform(0,1,len(final_survey)) <= 0.75
report = pd.concat([final_survey[features],x],axis = 1, ignore_index = True)
report_columns = features + numerical_inputs
report.columns = report_columns
report['survey_participant'] = final_survey['survey_participant'] 


# In[363]:

sum(report['survey_participant'])


# In[364]:

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler

le = sklearn.preprocessing.LabelEncoder()
nm = sklearn.preprocessing.StandardScaler()
variables = pd.DataFrame()

for column in features: 
    #print column
    x = report[column]
    if column == 'survey_participant': 
        continue
    if column not in numerical_inputs: 
        le.fit(x)
        var = pd.DataFrame(le.transform(x))
        variables = pd.concat([variables,var], axis=1)
    else:    
        nm.fit(x)
        var = pd.DataFrame(nm.transform(x))
        variables = pd.concat([variables,var], axis=1)


# In[365]:

variables.dropna()
variables.shape


# In[366]:

from sklearn.metrics import precision_recall_fscore_support as pr
def precisionRecall(x,y):
    # x is true positive 
    # x - y = 1 is false negative 
    # y - x = 1 is false positive 
    tp = x 
    fp = [sum(z)  for z in y-x if z>0]
    sum_y = sum(x)
    sum_yhat = sum(y)
    print sum_y/float(sum_y + sum_yhat)
    
    
    


# In[417]:

from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import GradientBoostingClassifier

gdc = GradientBoostingClassifier()
lr = LogisticRegression()
clf = svm.SVR()
et = ExtraTreesClassifier()
rgr = RadiusNeighborsRegressor()
forest = RandomForestRegressor(n_estimators = 100, n_jobs = 2, oob_score=True)
adaboost = AdaBoostRegressor()
nb = GaussianNB()
rd = RidgeClassifierCV()
kf = KFold(report.shape[0], n_folds = 5)

for train_index, test_index in kf:
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = variables.ix[list(train_index),], variables.ix[list(test_index),]
    y_train = report['survey_participant'].ix[list(train_index),]
    y_test = report['survey_participant'].ix[list(test_index),]
    forest.fit(X_train,y_train)
    adaboost.fit(X_train,y_train)
    gdc.fit(X_train, y_train)
    rd.fit(X_train, y_train)
    rgr.fit(X_train, y_train)
    nb.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    et.fit(X_train, y_train)
    #print forest.feature_importances_
    y_hat = list(gdc.predict(X_test))
    print 'GDC', sum((y_hat-y_test)**2)/float(len(y_test))
    y_hat = list(rd.predict(X_test))
    print 'RD', sum((y_hat-y_test)**2)/float(len(y_test))
    y_hat = list(et.predict(X_test))
    print 'ET', sum((y_hat-y_test)**2)/float(len(y_test))
    y_hat = list(lr.predict(X_test))
    print 'LR', sum((y_hat-y_test)**2)/float(len(y_test))
    y_hat = list(forest.predict(X_test))
    print 'RFRegressor', sum(((y_hat)-y_test)**2)/float(len(y_test))
   


# In[327]:

x = y_test
y = y_hat
tp = x 
fp = [sum(z)  for z in y-x if z>0]
sum_y = sum(x)
sum_yhat = sum(y)
print y_hat
#print sum_y/float(sum_y + sum_yhat)


# Now we break our data into testing and training sets. Later on, we shall deal with this by using a CV method 

# In[72]:

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(variables, report['survey_participant'][report['isTrain']== True])
y_hat = list(clf.predict(predictors))
sum((y_hat-y)**2)/float(len(y))


# In[101]:

# now that we have our data without NA values in report, let us vectorize the terms to use in our Random Forest 
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Imputer
forest = RandomForestClassifier(n_estimators = 100, n_jobs = 2,oob_score=True)

#,'q5_tip_ing_1_orig','q5_tip_ing_2_orig','q5_tip_ing_3_orig'
#pd.DataFrame(train[['q6_math_orig','q6_lang_orig']],x)

le = sklearn.preprocessing.LabelEncoder()
variables = pd.DataFrame()
predictors = pd.DataFrame()
for column in train: 
    x = report[column].dropna()
    le.fit(x)
    var = pd.DataFrame(le.transform(train[column].dropna()))
    variables = pd.concat([variables,var], axis=1)
    pred = pd.DataFrame(le.transform(test[column].dropna()))
    predictors = pd.concat([predictors,pred], axis=1)

#variables = pd.concat([variables,report[numerical]], axis = 1)

forest.fit(variables,report['survey_participant'][report['isTrain'] == True])

pred = pd.concat([pred1,pred2,pred3,pred4,pred5],axis=0)

y_hat = list(forest.predict(predictors))
y = report['survey_participant'][report['isTrain'] == False]
sum((y_hat-y)**2)


# In[ ]:




# In[ ]:




# In[165]:

reports = (survey[['search1_psu_math','search1_psu_lang','search1_area','search1_nivel','search1_carrer']]).dropna()
rows = list(reports.index)
len(rows)
dim = survey.shape
r = dim[0]
results = [0] * r 
for row in rows: 
    results[row] = 1
survey['survey_participant'] = results



# We can't use the mean strategy because it doesn't work well for something like missing wages, where there is a large variation based on factors like school type, career and level (making up data can be a bad choice because we often shift our data set towards what we want it to tell us - which may be different from what it actually tells us. 

# In[133]:

clf = RandomForestClassifier(n_estimators=100, criterion = 'entropy', max_features = 100)
y = survey['survey_participant']
x = survey.ix[:,3:37]

#scores = clf.fit(x,y)


# In[ ]:

for column in sparse_matrix: 
    print column


# In[ ]:




# In[ ]:

def cross_val(X, y, clf_class, **kwargs):
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_pred = y.copy()
   
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred
# Checks the accuracy of the algorithm
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# In[ ]:




# In[ ]:




# In[132]:

'''non parametric 
dimensionality reduction 
'''


# In[ ]:




# In[ ]:



