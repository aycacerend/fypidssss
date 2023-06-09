import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler


features = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds',
'is_host_login',
'is_guest_login',
'count',
'srv_count',
'serror_rate',
'srv_serror_rate',
'rerror_rate',
'srv_rerror_rate',
'same_srv_rate',
'diff_srv_rate',
'srv_diff_host_rate',
'dst_host_count',
'dst_host_srv_count',
'dst_host_same_srv_rate',
'dst_host_diff_srv_rate',
'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate',
'dst_host_serror_rate',
'dst_host_srv_serror_rate',
'dst_host_rerror_rate',
'dst_host_srv_rerror_rate',
'intrusion_type']

data = pd.read_csv('kddcup.data_10_percent_corrected', names=features, header=None)
data.head()
## reading is done

print('The number of data points are:',data.shape[0])
print('='*40)
print('The number of features are:',data.shape[1])
print('='*40)
print('Some of the features are:',features[:10])
output = data['intrusion_type'].values
labels = set(output)
print('The different type of output labels are:', labels)
print('=' * 100)
print('Number of different output labels are:', len(labels))

print('Null values in the dataset are: ',len(data[data.isnull().any(1)])) ##checking the nulls

duplicateRowsDF = data[data.duplicated()] ##checking duplicates

duplicateRowsDF.head(5)
data.drop_duplicates(subset=features, keep='first', inplace=True)
data.shape
data.to_pickle('data.pkl')
data = pd.read_pickle('data.pkl')

##class label for categories

plt.figure(figsize=(20,15))
class_distribution = data['intrusion_type'].value_counts()
class_distribution.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of yi in train data')
plt.grid()
plt.show()
sorted_yi = np.argsort(-class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', i+1, ':',class_distribution.values[i], '(', np.round((class_distribution.values[i]/data.shape[0]*100), 3), '%)')

##multiclassification

plt.figure(figsize=(20, 15))
sns.violinplot(x="intrusion_type", y="src_bytes", data=data)
plt.xticks(rotation=90)

##pairplot anaysis for bivariate

def pairplot(data, label, features=[]):
    '''
    This function creates pairplot taking 4 features from our dataset as default parameters along with the output variable
    '''
    sns.pairplot(data, hue=label, height=4, diag_kind='hist', vars=features,
            plot_kws={'alpha':0.6, 's':80, 'edgecolor':'k'})
    pairplot(data, 'intrusion_type', features=['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment'])
    pairplot(data, 'intrusion_type', features=['urgent', 'hot', 'num_failed_logins', 'num_compromised'])
    pairplot(data, 'intrusion_type', features=['root_shell', 'su_attempted', 'num_root', 'num_file_creations'])
    pairplot(data, 'intrusion_type', features=['num_shells', 'num_access_files', 'num_outbound_cmds', 'count'])
    pairplot(data, 'intrusion_type', features=['srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate'])
    pairplot(data, 'intrusion_type', features=['srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate'])
    pairplot(data, 'intrusion_type', features=['dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate'])
    pairplot(data, 'intrusion_type', features=['dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                       'dst_host_srv_serror_rate'])

##tsne for bivariate analysis
from datetime import datetime

df = data.drop(['intrusion_type','protocol_type','service','flag'], axis=1)
Y = data['intrusion_type'].values

from sklearn.manifold import TSNE


def tsne_func(data, label, no_components, perplexity_value, n_iter_value):

    start = datetime.now()
    print('TSNE with perplexity={} and no. of iterations={}'.format(perplexity_value, n_iter_value))
    tsne = TSNE(n_components=no_components, perplexity=perplexity_value, n_iter=n_iter_value)
    tsne_df1 = tsne.fit_transform(data)
    print(tsne_df1.shape)
    tsne_df1 = np.vstack((tsne_df1.T, Y)).T
    tsne_data1 = pd.DataFrame(data=tsne_df1, columns=['feature1', 'feature2', 'Output'])
    sns.FacetGrid(tsne_data1, hue='Output', size=6).map(plt.scatter, 'feature1', 'feature2').add_legend()
    print('Total time taken:',datetime.now()-start)
    plt.show()
    tsne_func(data=df, label=Y, no_components=2, perplexity_value=100, n_iter_value=500)##for 1
    tsne_func(data=df, label=Y, no_components=2, perplexity_value=50, n_iter_value=1000)##for 2



X_train, X_test, Y_train, Y_test = train_test_split(data.drop('intrusion_type', axis=1), data['intrusion_type'],
                                                    stratify=data['intrusion_type'], test_size=0.25)

print('Train data')
print(X_train.shape)
print(Y_train.shape)
print('=' * 20)
print('Test data')
print(X_test.shape)
print(Y_test.shape)
##one hot encoding for vectorizing

protocol = list(X_train['protocol_type'].values)
protocol = list(set(protocol))
print('Protocol types are:', protocol)

one_hot = CountVectorizer(vocabulary=protocol, binary=True)
train_protocol = one_hot.fit_transform(X_train['protocol_type'].values)
test_protocol = one_hot.transform(X_test['protocol_type'].values)

print(train_protocol[1].toarray())
train_protocol.shape
service = list(X_train['service'].values)
service = list(set(service))
print('Service types are:\n', service)

##from sklearn.feature_extraction.text import CountVectorizer
one_hot = CountVectorizer(vocabulary=service, binary=True)
train_service = one_hot.fit_transform(X_train['service'].values)
test_service = one_hot.transform(X_test['service'].values)
print(train_service[100].toarray())
train_service.shape
flag = list(X_train['flag'].values)
flag = list(set(flag))
print('Flag types are:', flag)
##from sklearn.feature_extraction.text import CountVectorizer

one_hot = CountVectorizer(binary=True)
one_hot.fit(X_train['flag'].values)
train_flag = one_hot.transform(X_train['flag'].values)
test_flag = one_hot.transform(X_test['flag'].values)

print(test_flag[3000].toarray())
train_flag.shape
X_train.drop(['protocol_type','service','flag'], axis=1, inplace=True)
X_test.drop(['protocol_type','service','flag'], axis=1, inplace=True)

##standardisation
def feature_scaling(X_train, X_test, feature_name):

    ##from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler1 = scaler.fit_transform(X_train[feature_name].values.reshape(-1, 1))
    scaler2 = scaler.transform(X_test[feature_name].values.reshape(-1, 1))

    return scaler1, scaler2


duration1, duration2 = feature_scaling(X_train, X_test, 'duration')

print(duration1[1])
src_bytes1, src_bytes2 = feature_scaling(X_train, X_test, 'src_bytes')

print(src_bytes1[1])
dst_bytes1, dst_bytes2 = feature_scaling(X_train, X_test, 'dst_bytes')

print(dst_bytes1[1])
wrong_fragment1, wrong_fragment2 = feature_scaling(X_train, X_test, 'wrong_fragment')

print(wrong_fragment1[1])
urgent1, urgent2 = feature_scaling(X_train, X_test, 'urgent')

print(urgent1[1])
hot1, hot2 = feature_scaling(X_train, X_test, 'hot')

print(hot1[1])

num_failed_logins1, num_failed_logins2 = feature_scaling(X_train, X_test, 'num_failed_logins')

print(num_failed_logins1[1])

num_compromised1, num_compromised2 = feature_scaling(X_train, X_test, 'num_compromised')

num_compromised1[1]

root_shell1, root_shell2 = feature_scaling(X_train, X_test, 'root_shell')

root_shell1[1]

su_attempted1, su_attempted2 = feature_scaling(X_train, X_test, 'su_attempted')

su_attempted1[1]

num_root1, num_root2 = feature_scaling(X_train, X_test, 'num_root')

num_root1[1]

num_file_creations1, num_file_creations2 = feature_scaling(X_train, X_test, 'num_file_creations')

num_file_creations1[1]
num_shells1, num_shells2 = feature_scaling(X_train, X_test, 'num_shells')

num_shells1[1]
num_access_files1, num_access_files2 = feature_scaling(X_train, X_test, 'num_access_files')

num_access_files1[1]

data['num_outbound_cmds'].value_counts()
srv_count1, srv_count2 = feature_scaling(X_train, X_test, 'srv_count')
srv_count1[1]
serror_rate1, serror_rate2 = feature_scaling(X_train, X_test, 'serror_rate')
serror_rate1[1]
srv_serror_rate1, srv_serror_rate2 = feature_scaling(X_train, X_test, 'srv_serror_rate')

srv_serror_rate1[1]
rerror_rate1, rerror_rate2 = feature_scaling(X_train, X_test, 'rerror_rate')

rerror_rate1[1]
srv_rerror_rate1, srv_rerror_rate2 = feature_scaling(X_train, X_test, 'srv_rerror_rate')

srv_rerror_rate1[1]
same_srv_rate1, same_srv_rate2 = feature_scaling(X_train, X_test, 'same_srv_rate')

same_srv_rate1[1]
diff_srv_rate1, diff_srv_rate2 = feature_scaling(X_train, X_test, 'diff_srv_rate')

diff_srv_rate1[1]
srv_diff_host_rate1, srv_diff_host_rate2 = feature_scaling(X_train, X_test, 'srv_diff_host_rate')

srv_diff_host_rate1[1]
dst_host_count1, dst_host_count2 = feature_scaling(X_train, X_test, 'dst_host_count')

dst_host_count1[1]
dst_host_srv_count1, dst_host_srv_count2 = feature_scaling(X_train, X_test, 'dst_host_srv_count')

dst_host_srv_count1[1]
dst_host_same_srv_rate1, dst_host_same_srv_rate2= feature_scaling(X_train, X_test, 'dst_host_same_srv_rate')

dst_host_same_srv_rate1[1]
dst_host_diff_srv_rate1, dst_host_diff_srv_rate2 = feature_scaling(X_train, X_test, 'dst_host_diff_srv_rate')

dst_host_diff_srv_rate1[1]
dst_host_same_src_port_rate1, dst_host_same_src_port_rate2 = feature_scaling(X_train, X_test,
                                                                             'dst_host_same_src_port_rate')

dst_host_same_src_port_rate1[1]
dst_host_srv_diff_host_rate1, dst_host_srv_diff_host_rate2 = feature_scaling(X_train, X_test,
                                                                             'dst_host_srv_diff_host_rate')

dst_host_srv_diff_host_rate1[1]
dst_host_serror_rate1, dst_host_serror_rate2 = feature_scaling(X_train, X_test, 'dst_host_serror_rate')

dst_host_serror_rate1[1]
dst_host_srv_serror_rate1, dst_host_srv_serror_rate2 = feature_scaling(X_train, X_test, 'dst_host_srv_serror_rate')

dst_host_srv_serror_rate1[1]
dst_host_rerror_rate1, dst_host_rerror_rate2 = feature_scaling(X_train, X_test, 'dst_host_rerror_rate')

dst_host_rerror_rate1[1]
dst_host_srv_rerror_rate1, dst_host_srv_rerror_rate2 = feature_scaling(X_train, X_test, 'dst_host_srv_rerror_rate')

dst_host_srv_rerror_rate1[1]
num_failed_logins1, num_failed_logins2 = feature_scaling(X_train, X_test, 'num_failed_logins')

num_failed_logins1[1]
land1, land2 = np.array([X_train['land'].values]), np.array([X_test['land'].values])

land1.shape
is_host_login1, is_host_login2 = np.array([X_train['is_host_login'].values]), np.array([X_test['is_host_login'].values])

is_host_login1.shape
is_guest_login1, is_guest_login2 = np.array([X_train['is_guest_login'].values]), np.array(
    [X_test['is_guest_login'].values])

is_guest_login1.shape
logged_in1, logged_in2 = np.array([X_train['logged_in'].values]), np.array([X_test['logged_in'].values])

logged_in1.shape
count1, count2 = feature_scaling(X_train, X_test, 'count')

count1[1]
dst_host_diff_srv_rate1, dst_host_diff_srv_rate2 = feature_scaling(X_train, X_test, 'dst_host_diff_srv_rate')

dst_host_diff_srv_rate1[1]

##merging
from scipy.sparse import hstack

X_train_1 = hstack((duration1, train_protocol, train_service, train_flag, src_bytes1,
                    dst_bytes1, land1.T, wrong_fragment1, urgent1, hot1,
                    num_failed_logins1, logged_in1.T, num_compromised1, root_shell1,
                    su_attempted1, num_root1, num_file_creations1, num_shells1,
                    num_access_files1, is_host_login1.T,
                    is_guest_login1.T, count1, srv_count1, serror_rate1,
                    srv_serror_rate1, rerror_rate1, srv_rerror_rate1, same_srv_rate1,
                    diff_srv_rate1, srv_diff_host_rate1, dst_host_count1,
                    dst_host_srv_count1, dst_host_same_srv_rate1,
                    dst_host_diff_srv_rate1, dst_host_same_src_port_rate1,
                    dst_host_srv_diff_host_rate1, dst_host_serror_rate1,
                    dst_host_srv_serror_rate1, dst_host_rerror_rate1,
                    dst_host_srv_rerror_rate1))

X_train_1.shape
X_test_1 = hstack((duration2, test_protocol, test_service, test_flag, src_bytes2,
                   dst_bytes2, land2.T, wrong_fragment2, urgent2, hot2,
                   num_failed_logins2, logged_in2.T, num_compromised2, root_shell2,
                   su_attempted2, num_root2, num_file_creations2, num_shells2,
                   num_access_files2, is_host_login2.T,
                   is_guest_login2.T, count2, srv_count2, serror_rate2,
                   srv_serror_rate2, rerror_rate2, srv_rerror_rate2, same_srv_rate2,
                   diff_srv_rate2, srv_diff_host_rate2, dst_host_count2,
                   dst_host_srv_count2, dst_host_same_srv_rate2,
                   dst_host_diff_srv_rate2, dst_host_same_src_port_rate2,
                   dst_host_srv_diff_host_rate2, dst_host_serror_rate2,
                   dst_host_srv_serror_rate2, dst_host_rerror_rate2,
                   dst_host_srv_rerror_rate2))

X_test_1.shape
import joblib

joblib.dump(X_train_1, 'X_train_1.pkl')
joblib.dump(X_test_1, 'X_test_1.pkl')
X_train_1 = joblib.load('X_train_1.pkl')
X_test_1 = joblib.load('X_test_1.pkl')

joblib.dump(Y_train, 'Y_train.pkl')
joblib.dump(Y_test, 'Y_test.pkl')
Y_train = joblib.load('Y_train.pkl')
Y_test = joblib.load('Y_test.pkl')


##Applying Algorithms

import datetime as dt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib


def confusion_matrix_func(Y_test, y_test_pred):

    C = confusion_matrix(Y_test, y_test_pred)
    cm_df = pd.DataFrame(C)
    labels = ['back', 'butter_overflow', 'loadmodule', 'guess_passwd', 'imap', 'ipsweep', 'warezmaster', 'rootkit',
              'multihop', 'neptune', 'nmap', 'normal', 'phf', 'perl', 'pod', 'portsweep', 'ftp_write', 'satan', 'smurf',
              'teardrop', 'warezclient', 'land']
    plt.figure(figsize=(20, 15))
    sns.set(font_scale=1.4)
    sns.heatmap(cm_df, annot=True, annot_kws={"size": 12}, fmt='g', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')

    plt.show()


def model(model_name, X_train, Y_train, X_test, Y_test):
    print('Fitting the model and prediction on train data:')
    start = dt.datetime.now()
    model_name.fit(X_train, Y_train)
    y_tr_pred = model_name.predict(X_train)
    print('Completed')
    print('Time taken:', dt.datetime.now() - start)
    print('=' * 50)

    results_tr = dict()
    y_tr_pred = model_name.predict(X_train)
    results_tr['precision'] = precision_score(Y_train, y_tr_pred, average='weighted')
    results_tr['recall'] = recall_score(Y_train, y_tr_pred, average='weighted')
    results_tr['f1_score'] = f1_score(Y_train, y_tr_pred, average='weighted')

    results_test = dict()
    print('Prediction on test data:')
    start = dt.datetime.now()
    y_test_pred = model_name.predict(X_test)
    print('Completed')
    print('Time taken:', dt.datetime.now() - start)
    print('=' * 50)

    print('Performance metrics:')
    print('=' * 50)
    print('Confusion Matrix is:')
    confusion_matrix_func(Y_test, y_test_pred)
    print('=' * 50)
    results_test['precision'] = precision_score(Y_test, y_test_pred, average='weighted')
    print('Precision score is:')
    print(precision_score(Y_test, y_test_pred, average='weighted'))
    print('=' * 50)
    results_test['recall'] = recall_score(Y_test, y_test_pred, average='weighted')
    print('Recall score is:')
    print(recall_score(Y_test, y_test_pred, average='weighted'))
    print('=' * 50)
    results_test['f1_score'] = f1_score(Y_test, y_test_pred, average='weighted')
    print('F1-score is:')
    print(f1_score(Y_test, y_test_pred, average='weighted'))
    results_test['model'] = model

    return results_tr, results_test


def print_grid_search_attributes(model):

    print('---------------------------')
    print('|      Best Estimator     |')
    print('---------------------------')
    print('\n\t{}\n'.format(model.best_estimator_))

    # parameters that gave best results while performing grid search
    print('---------------------------')
    print('|     Best parameters     |')
    print('---------------------------')
    print('\tParameters of best estimator : \n\n\t{}\n'.format(model.best_params_))

    #  number of cross validation splits
    print('----------------------------------')
    print('|   Number of CrossValidation sets   |')
    print('----------------------------------')
    print('\n\tTotal number of cross validation sets: {}\n'.format(model.n_splits_))

    # Average cross validated score of the best estimator, from the Grid Search
    print('---------------------------')
    print('|        Best Score       |')
    print('---------------------------')
    print('\n\tAverage Cross Validate scores of best estimator : \n\n\t{}\n'.format(model.best_score_))

##computes tpr and fpr scores
def tpr_fpr_func(Y_tr, Y_pred):

        results = dict()
        Y_tr = Y_tr.to_list()
        tp = 0;
        fp = 0;
        positives = 0;
        negatives = 0;
        length = len(Y_tr)
        for i in range(len(Y_tr)):
            if Y_tr[i] == 'normal.':
                positives += 1
            else:
                negatives += 1

        for i in range(len(Y_pred)):
            if Y_tr[i] == 'normal.' and Y_pred[i] == 'normal.':
                tp += 1
            elif Y_tr[i] != 'normal.' and Y_pred[i] == 'normal.':
                fp += 1

        tpr = tp / positives
        fpr = fp / negatives

        results['tp'] = tp;
        results['tpr'] = tpr;
        results['fp'] = fp;
        results['fpr'] = fpr

        return results
        ##naive bayers
hyperparameter = {'var_smoothing': [10 ** x for x in range(-9, 3)]}

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb_grid = GridSearchCV(nb, param_grid=hyperparameter, cv=5, verbose=1, n_jobs=-1)
nb_grid_results_tr, nb_grid_results_test = model(nb_grid, X_train_1.toarray(), Y_train, X_test_1.toarray(), Y_test)
print_grid_search_attributes(nb_grid)
joblib.dump(nb_grid.best_estimator_, 'nb_gs.pkl')
nb_gs = nb_grid.best_estimator_
y_tr_pred = nb_gs.predict(X_train_1.toarray())
y_test_pred = nb_gs.predict(X_test_1.toarray())
tpr_fpr_train = tpr_fpr_func(Y_train, y_tr_pred)
tpr_fpr_test = tpr_fpr_func(Y_test, y_test_pred)
tpr_fpr_train
nb_grid_results_tr
tpr_fpr_test
nb_grid_results_test

##logistic regression
hyperparameter = {'alpha':[0.001, 0.01, 0.1, 1, 10, 20, 30], 'penalty':['l1', 'l2']}
from sklearn.linear_model import SGDClassifier
lr = SGDClassifier(loss='log_loss')
lr_grid = GridSearchCV(lr, param_grid=hyperparameter, cv=5, verbose=1, n_jobs=-1)
lr_grid_results_tr, lr_grid_results_test = model(lr_grid, X_train_1.toarray(), Y_train, X_test_1.toarray(), Y_test)
print_grid_search_attributes(lr_grid)
joblib.dump(lr_grid.best_estimator_, 'lr_gs.pkl')
lr_gs = lr_grid.best_estimator_
y_tr_pred = lr_gs.predict(X_train_1.toarray())
y_test_pred = lr_gs.predict(X_test_1.toarray())
tpr_fpr_train = tpr_fpr_func(Y_train, y_tr_pred)
tpr_fpr_test = tpr_fpr_func(Y_test, y_test_pred)

lr_grid_results_tr
tpr_fpr_train
lr_grid_results_test
tpr_fpr_test

##svm
hyperparameter = {'alpha': [10 ** x for x in range(-8, 3)], 'penalty': ['l1', 'l2']}

from sklearn.linear_model import SGDClassifier

svm = SGDClassifier(loss='hinge')
svm_grid = GridSearchCV(svm, param_grid=hyperparameter, cv=5, verbose=1, n_jobs=-1)

svm_grid_results_tr, svm_grid_results_test = model(svm_grid, X_train_1.toarray(), Y_train, X_test_1.toarray(),
                                                   Y_test)
print_grid_search_attributes(svm_grid)
joblib.dump(svm_grid.best_estimator_, 'svm_gs.pkl')
svm_gs = svm_grid.best_estimator_
y_tr_pred = svm_gs.predict(X_train_1.toarray())
y_test_pred = svm_gs.predict(X_test_1.toarray())
svm_tpr_fpr_train = tpr_fpr_func(Y_train, y_tr_pred)
svm_tpr_fpr_test = tpr_fpr_func(Y_test, y_test_pred)

svm_grid_results_tr
svm_tpr_fpr_train
svm_grid_results_test
svm_tpr_fpr_test


hyperparameter = {'max_depth': [5, 10, 20, 50, 100, 500], 'min_samples_split': [5, 10, 100, 500]}

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(criterion='gini', splitter='best', class_weight='balanced')
decision_tree_grid = GridSearchCV(decision_tree, param_grid=hyperparameter, cv=3, verbose=1, n_jobs=-1)

decision_tree_grid_results_tr, decision_tree_grid_results_test = model(decision_tree_grid, X_train_1.toarray(),
                                                                       Y_train, X_test_1.toarray(), Y_test)

print_grid_search_attributes(decision_tree_grid)
joblib.dump(decision_tree_grid.best_estimator_, 'decision_tree_gs.pkl')
dt_gs = decision_tree_grid.best_estimator_
y_tr_pred = dt_gs.predict(X_train_1.toarray())
y_test_pred = dt_gs.predict(X_test_1.toarray())
dt_tpr_fpr_train = tpr_fpr_func(Y_train, y_tr_pred)
dt_tpr_fpr_test = tpr_fpr_func(Y_test, y_test_pred)

decision_tree_grid_results_tr
dt_tpr_fpr_train
decision_tree_grid_results_test
dt_tpr_fpr_test


hyperparameter = {'max_depth': [5, 10, 100, 500, 1000], 'n_estimators': [5, 10, 50,  100, 500],
                          'min_samples_split': [5, 10, 100, 500]}
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion='gini', class_weight='balanced')
rf_grid = GridSearchCV(rf, param_grid=hyperparameter, cv=3, verbose=1, n_jobs=-1)

rf_grid_results_tr, rf_grid_results_test = model(rf_grid, X_train_1.toarray(), Y_train, X_test_1.toarray(), Y_test)
print_grid_search_attributes(rf_grid)
joblib.dump(rf_grid.best_estimator_, 'rf_gs.pkl')
rf_gs = rf_grid.best_estimator_
y_tr_pred = rf_gs.predict(X_train_1.toarray())
y_test_pred = rf_gs.predict(X_test_1.toarray())
rf_tpr_fpr_train = tpr_fpr_func(Y_train, y_tr_pred)
rf_tpr_fpr_test = tpr_fpr_func(Y_test, y_test_pred)
rf_grid_results_tr
rf_tpr_fpr_train
rf_grid_results_test
rf_tpr_fpr_test


##xg boost

hyperparameter = {'max_depth': [2, 3, 5, 7, 10], 'n_estimators': [10, 50, 100, 200, 500]}

from xgboost import XGBClassifier

xgb = XGBClassifier(objective='multi:softprob')
xgb_grid = RandomizedSearchCV(xgb, param_distributions=hyperparameter, cv=3, verbose=1, n_jobs=-1)
joblib.dump(xgb_grid.best_estimator_, 'xgb_gs.pkl')
xgb_best = xgb_grid.best_estimator_
xgb_grid_results_tr, xgb_grid_results_test = model(xgb_best, X_train_1.toarray(), Y_train, X_test_1.toarray(),
                                                   Y_test)
print_grid_search_attributes(xgb_grid)

xgb_gs = xgb_grid.best_estimator_
y_tr_pred = xgb_gs.predict(X_train_1.toarray())
y_test_pred = xgb_gs.predict(X_test_1.toarray())
xgb_tpr_fpr_train = tpr_fpr_func(Y_train, y_tr_pred)
xgb_tpr_fpr_test = tpr_fpr_func(Y_test, y_test_pred)
xgb_grid_results_tr
xgb_tpr_fpr_train
xgb_grid_results_test
xgb_tpr_fpr_test


##adding features
data = pd.read_pickle('data.pkl')
print('Shape of our dataset', data.shape)
from sklearn.cluster import MiniBatchKMeans
import numpy as np
kmeans_1 = MiniBatchKMeans(n_clusters=23, random_state=0, batch_size=128,
                                   max_iter=1000000)  # Multiclass classification
kmeans_1.fit(X_train_1)
train_cluster_1 = kmeans_1.predict(X_train_1)
test_cluster_1 = kmeans_1.predict(X_test_1)
print('Length of train cluster is:', len(train_cluster_1))
train_cluster_1
train_cluster_1 = np.array([train_cluster_1])
train_cluster_1.shape
print('Length of test cluster is', len(test_cluster_1))
test_cluster_1
test_cluster_1 = np.array([test_cluster_1])
test_cluster_1.shape
kmeans_2 = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=128, max_iter=1000000)  # Binary Classification
kmeans_2.fit(X_train_1)
train_cluster_2 = kmeans_2.predict(X_train_1)
test_cluster_2 = kmeans_2.predict(X_test_1)
print('Length of train cluster is:', len(train_cluster_2))
train_cluster_2
print('Length of test cluster is', len(test_cluster_2))
test_cluster_2
train_cluster_2 = np.array([train_cluster_2])
train_cluster_2.shape
test_cluster_2 = np.array([test_cluster_2])
test_cluster_2.shape


from sklearn.decomposition import PCA

pca = PCA(n_components=5)
pca.fit(X_train_1.toarray())
pca_train = pca.transform(X_train_1.toarray())
pca_test = pca.transform(X_test_1.toarray())
print(pca_train.shape)
print(pca_test.shape)
feature_src_dst_1 = src_bytes1 + dst_bytes1
feature_src_dst_2 = src_bytes2 + dst_bytes2
feature_src_dst_1.shape
feature_src_dst_3 = abs(src_bytes1 - dst_bytes1)
feature_src_dst_4 = abs(src_bytes2 - dst_bytes2)
feature_src_dst_3.shape
feature_5 = same_srv_rate1 + diff_srv_rate1
feature_6 = same_srv_rate2 + diff_srv_rate2
feature_5.shape
feature_7 = dst_host_same_srv_rate1 + dst_host_diff_srv_rate1
feature_8 = dst_host_same_srv_rate2 + dst_host_diff_srv_rate2
feature_7.shape

# adding cluster and pca to dataset

X_train_2 = hstack((X_train_1, pca_train, train_cluster_1.T, train_cluster_2.T, feature_src_dst_1,
                    feature_src_dst_3, feature_5, feature_7))
X_test_2 = hstack((X_test_1, pca_test, test_cluster_1.T, test_cluster_2.T, feature_src_dst_2, feature_src_dst_4,
                   feature_6, feature_8))

print('Train data:')
print(X_train_2.shape)
print('=' * 30)
print('Test data:')
print(X_test_2.shape)
joblib.dump(X_train_2, 'X_train_2.pkl')
joblib.dump(X_test_2, 'X_test_2.pkl')

#for decision tree
hyperparameter = {'max_depth': [5, 10, 20, 50, 100, 500], 'min_samples_split': [5, 10, 100, 500]}
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(criterion='gini', splitter='best', class_weight='balanced')
decision_tree_grid = GridSearchCV(decision_tree, param_grid=hyperparameter, cv=5, verbose=1, n_jobs=-1)
decision_tree_results2_tr, decision_tree_results2_test = model(decision_tree_grid, X_train_2.toarray(), Y_train,
                                                               X_test_2.toarray(), Y_test)
print_grid_search_attributes(decision_tree_grid)
dt_gs = decision_tree_grid.best_estimator_
y_tr_pred = dt_gs.predict(X_train_2.toarray())
y_test_pred = dt_gs.predict(X_test_2.toarray())
dt_tpr_fpr_train = tpr_fpr_func(Y_train, y_tr_pred)

dt_tpr_fpr_test = tpr_fpr_func(Y_test, y_test_pred)
decision_tree_results2_tr
dt_tpr_fpr_train
decision_tree_results2_test
dt_tpr_fpr_test
joblib.dump(decision_tree_grid.best_estimator_, 'dt2.pkl')


#for random forest

hyperparameter = {'max_depth': [5, 10, 100, 500, 1000], 'n_estimators': [5, 10, 50, 100, 500],
                  'min_samples_split': [5, 10, 100, 500]}

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='gini', class_weight='balanced')
rf_grid = GridSearchCV(rf, param_grid=hyperparameter, cv=3, verbose=1, n_jobs=-1)
rf_grid_results_tr, rf_grid_results_test = model(rf_grid, X_train_2.toarray(), Y_train, X_test_2.toarray(),
                                                 Y_test)
print_grid_search_attributes(rf_grid)
rf_gs = rf_grid.best_estimator_
y_tr_pred = rf_gs.predict(X_train_2.toarray())
y_test_pred = rf_gs.predict(X_test_2.toarray())
rf_tpr_fpr_train = tpr_fpr_func(Y_train, y_tr_pred)
rf_tpr_fpr_test = tpr_fpr_func(Y_test, y_test_pred)
rf_grid_results_tr
rf_tpr_fpr_train
rf_grid_results_test
rf_tpr_fpr_test
joblib.dump(rf_grid.best_estimator_, 'rf2.pkl')

#for xgboost
hyperparameter = {'max_depth': [2, 3, 5, 7, 10], 'n_estimators': [10, 50, 100, 200, 500]}
from xgboost import XGBClassifier

xgb = XGBClassifier(objective='multi:softprob', n_jobs=-1)
xgb_grid = RandomizedSearchCV(xgb, param_distributions=hyperparameter, cv=3, verbose=1, n_jobs=-1)
xgb_grid_results_tr, xgb_grid_results_test = model(xgb_grid, X_train_2.toarray(), Y_train, X_test_2.toarray(),
                                                   Y_test)

print_grid_search_attributes(xgb_grid)
xgb_gs = xgb_grid.best_estimator_
y_tr_pred = xgb_gs.predict(X_train_2.toarray())
y_test_pred = xgb_gs.predict(X_test_2.toarray())
xgb_tpr_fpr_train = tpr_fpr_func(Y_train, y_tr_pred)
xgb_tpr_fpr_test = tpr_fpr_func(Y_test, y_test_pred)
xgb_grid_results_tr
xgb_tpr_fpr_train
xgb_grid_results_test
xgb_tpr_fpr_test
joblib.dump(xgb_grid.best_estimator_, 'xgb2.pkl')


##summarize the table
from prettytable import PrettyTable

x = PrettyTable()

x.field_names = ['Model', 'Train f1-score', 'Train TPR', 'Train FPR', 'Test f1-score', 'Test TPR', 'Test FPR']
x.add_row(['Naive Bayes', '0.9671', '99.40%', '5.13%', '0.9679', '99.34%', '4.91%'])
x.add_row(['Logistic Regression', '0.9813', '99.81%', '2.95%', '0.9819', '99.81%', '2.76%'])
x.add_row(['Support Vector Machine', '0.9967', '99.87%', '0.48%', '0.9966', '99.87%', '0.43%'])
x.add_row(['Decision Tree - 1', '0.9997', '99.96%', '0.0%', '0.9986', '99.90%', '0.13%'])
x.add_row(['Random Forest - 1', '0.9999', '99.98%', '0.0%', '0.9992', '99.98%', '0.13%'])
x.add_row(['XG Boost - 1', '0.9999', '100.0%', '0.0%', '0.9994', '99.98%', '0.083%'])
x.add_row(['Decision Tree - 2', '0.9998', '99.97%', '0.0%', '0.9986', '99.89%', '0.09%'])
x.add_row(['Random Forest - 2', '0.9999', '99.99%', '0.0%', '0.9990', '99.99%', '0.15%'])
x.add_row(['XG Boost - 2', '0.9999', '99.99%', '0.0%', '0.9994', '99.98%', '0.083%'])

print(x)



##binary classification
data.head()
intrusion_binary = []
for i in data['intrusion_type'].values:
    if i == 'normal.':
        intrusion_binary.append(1)
    else:
        intrusion_binary.append(0)
print(len(intrusion_binary))
print(intrusion_binary[:10])
data['intrusion_binary'] = intrusion_binary
data.head()
data.drop('intrusion_type', axis=1, inplace=True)
print('Shape of the data is:')
print(data.shape)
print('=' * 80)
print('Features of the dataset:')
print(data.columns)


def pairplot(data, label, features=[]):
    '''
    This function creates pairplot taking 4 features from our dataset as default parameters along with the output variable
    '''
    sns.pairplot(data, hue=label, height=4, diag_kind='hist', vars=features,
                 plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'})
    pairplot(data, 'intrusion_binary', features=['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment'])
    pairplot(data, 'intrusion_binary', features=['urgent', 'hot', 'num_failed_logins', 'num_compromised'])
    pairplot(data, 'intrusion_binary',
             features=['root_shell', 'su_attempted', 'num_root', 'num_file_creations'])
    pairplot(data, 'intrusion_binary',
             features=['num_shells', 'num_access_files', 'num_outbound_cmds', 'count'])
    pairplot(data, 'intrusion_binary', features=['srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate'])
    pairplot(data, 'intrusion_binary',
             features=['srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate'])
    pairplot(data, 'intrusion_binary',
             features=['dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                       'dst_host_srv_serror_rate'])
    pairplot(data, 'intrusion_binary',
             features=['dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                       'dst_host_diff_srv_rate'])


def tsne_func(data, label, no_components, perplexity_value, n_iter_value):
        '''
        This function applies TSNE on the original dataset with no_components, perplexity_value, n_iter_value as the TSNE
        parameters and transforms the original dataset into TSNE transformed feature space with the tsne dataset containing
        number of features equal to the value specified for no_components and also plots the scatter plot of the transformed
        data points along with their class label
        '''
        from datetime import datetime
        from sklearn.manifold import TSNE

        df = data.drop(['intrusion_type', 'protocol_type', 'service', 'flag'], axis=1)
        Y = data['intrusion_type'].values
        start = datetime.now()
        print('TSNE with perplexity={} and no. of iterations={}'.format(perplexity_value, n_iter_value))
        tsne = TSNE(n_components=no_components, perplexity=perplexity_value, n_iter=n_iter_value)
        tsne_df1 = tsne.fit_transform(data)
        print(tsne_df1.shape)
        tsne_df1 = np.vstack((tsne_df1.T, Y)).T
        tsne_data1 = pd.DataFrame(data=tsne_df1, columns=['feature1', 'feature2', 'Output'])
        sns.FacetGrid(tsne_data1, hue='Output', size=6).map(plt.scatter, 'feature1', 'feature2').add_legend()
        print('Total time taken:', datetime.now() - start)
        plt.show()
        df = data.drop(['intrusion_binary', 'service', 'flag', 'protocol_type'], axis=1)
        Y = data['intrusion_binary'].values

        tsne_func(data=df, label=Y, no_components=2, perplexity_value=100, n_iter_value=500)
        tsne_func(data=df, label=Y, no_components=2, perplexity_value=50, n_iter_value=1000)

X_train_3 = hstack(
    (X_train_1, pca_train, train_cluster_2.T, feature_src_dst_1, feature_src_dst_3, feature_5, feature_7))
X_test_3 = hstack(
    (X_test_1, pca_test, test_cluster_2.T, feature_src_dst_2, feature_src_dst_4, feature_6, feature_8))
print('Train data:')
print(X_train_3.shape)
print('=' * 30)
print('Test data:')
print(X_test_3.shape)
joblib.dump(X_train_3, 'X_train_3.pkl')
joblib.dump(X_test_3, 'X_test_3.pkl')
Y_train_new = []
for i in Y_train:
    if i == 'normal.':
        Y_train_new.append(1)
    else:
        Y_train_new.append(0)
print(len(Y_train_new))

Y_train_new[:5]
Y_test_new = []

for i in Y_test:
    if i == 'normal.':
        Y_test_new.append(1)
    else:
        Y_test_new.append(0)
print(len(Y_test_new))
Y_test_new[:5]
joblib.dump(Y_train_new, 'Y_train_new.pkl')
joblib.dump(Y_test_new, 'Y_test_new.pkl')

##applying machine learning models
from sklearn.metrics import auc
def confusion_matrix_func(Y_test, y_test_pred):
            '''
            This function plots the confusion matrix for Binary classification problem
            '''
            C = confusion_matrix(Y_test, y_test_pred)
            cm_df = pd.DataFrame(C)
            labels = ['BAD', 'NORMAL']
            plt.figure(figsize=(5, 4))
            sns.set(font_scale=1.4)
            sns.heatmap(cm_df, annot=True, annot_kws={"size": 12}, fmt='g', xticklabels=labels, yticklabels=labels)
            plt.ylabel('Actual Class')
            plt.xlabel('Predicted Class')

            plt.show()




def tpr_fpr_func_2(Y_tr, Y_pred):
        '''
                        This function computes the TPR and FPR using the actual and predicted values for each of the models.
                        '''
        results = dict()
        # Y_tr = Y_tr.to_list()
        tp = 0;
        fp = 0;
        positives = 0;
        negatives = 0;
        length = len(Y_tr)
        for i in range(len(Y_tr)):
            if Y_tr[i] == 1:
                positives += 1
            elif Y_tr[i] == 0:
                negatives += 1

        for i in range(len(Y_pred)):
            if Y_tr[i] == 1 and Y_pred[i] == 1:
                tp += 1
            elif Y_tr[i] == 0 and Y_pred[i] == 1:
                fp += 1

        tpr = tp / (positives)
        fpr = fp / (negatives)

        results['tp'] = tp;
        results['tpr'] = tpr;
        results['fp'] = fp;
        results['fpr'] = fpr

        return results

def model(model_name, X_train, Y_train, X_test, Y_test):
            '''
            This function computes the performance metric scores on the train and test data.
            '''

            print('Fitting the model and prediction on train data:')
            start = dt.datetime.now()
            model_name.fit(X_train, Y_train)
            y_tr_pred = model_name.predict(X_train)
            print('Completed')
            print('Time taken:', dt.datetime.now() - start)
            print('=' * 50)

            results_tr = dict()
            y_tr_pred = model_name.predict(X_train)
            results_tr['precision'] = precision_score(Y_train, y_tr_pred, average='weighted')
            results_tr['recall'] = recall_score(Y_train, y_tr_pred, average='weighted')
            results_tr['f1_score'] = f1_score(Y_train, y_tr_pred, average='weighted')

            results_test = dict()
            print('Prediction on test data:')
            start = dt.datetime.now()
            y_test_pred = model_name.predict(X_test)
            print('Completed')
            print('Time taken:', dt.datetime.now() - start)
            print('=' * 50)
            print('Train Confusion Matrix is:')
            confusion_matrix_func(Y_train, y_tr_pred)
            print('=' * 50)
            print('F1-score is:')
            print(f1_score(Y_test, y_test_pred, average='weighted'))
            print('=' * 50)

            print('Test Performance metrics:')
            print('=' * 50)
            print('Confusion Matrix is:')
            confusion_matrix_func(Y_test, y_test_pred)
            print('=' * 50)
            results_test['precision'] = precision_score(Y_test, y_test_pred, average='weighted')
            print('Precision score is:')
            print(precision_score(Y_test, y_test_pred, average='weighted'))
            print('=' * 50)
            results_test['recall'] = recall_score(Y_test, y_test_pred, average='weighted')
            print('Recall score is:')
            print(recall_score(Y_test, y_test_pred, average='weighted'))
            print('=' * 50)
            results_test['f1_score'] = f1_score(Y_test, y_test_pred, average='weighted')
            print('F1-score is:')
            print(f1_score(Y_test, y_test_pred, average='weighted'))
            # add the trained  model to the results
            results_test['model'] = model

            return results_tr, results_test


hyperparameter = {'max_depth': [5, 10, 20, 50, 100, 500], 'min_samples_split': [5, 10, 100, 500]}

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(criterion='gini', splitter='best', class_weight='balanced')
decision_tree_grid = GridSearchCV(decision_tree, param_grid=hyperparameter, cv=5, verbose=1, n_jobs=-1)

decision_tree_grid_results_tr, decision_tree_grid_results_test = model(decision_tree_grid, X_train_3.toarray(),
                                                                       Y_train_new, X_test_3.toarray(),
                                                                       Y_test_new)

dt_gs = decision_tree_grid.best_estimator_
y_tr_pred = dt_gs.predict(X_train_3.toarray())
y_test_pred = dt_gs.predict(X_test_3.toarray())
dt_tpr_fpr_train = tpr_fpr_func_2(Y_train_new, y_tr_pred)
dt_tpr_fpr_test = tpr_fpr_func_2(Y_test_new, y_test_pred)
dt_tpr_fpr_train
dt_tpr_fpr_test
print_grid_search_attributes(decision_tree_grid)

# https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
from sklearn.metrics import roc_curve

clf = decision_tree_grid.best_estimator_
clf.fit(X_train_3, Y_train_new)
Y_1 = clf.predict_proba(X_train_3)
Y1 = Y_1[:, 1]
train_fpr, train_tpr, tr_threshold = roc_curve(Y_train_new, Y1)
Y_2 = clf.predict_proba(X_test_3)
Y2 = Y_2[:, 1]
test_fpr, test_tpr, te_threshold = roc_curve(Y_test_new, Y2)
plt.plot(train_fpr, train_tpr, label='Train_AUC = ' + str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label='Test_AUC = ' + str(auc(test_fpr, test_tpr)))
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.title('AUC')
plt.legend()
plt.grid()
plt.show()
decision_tree_grid_results_tr

decision_tree_grid_results_test
joblib.dump(decision_tree_grid.best_estimator_, 'dt_grid_3.pkl')

hyperparameter = {'max_depth': [5, 10, 100, 500, 1000], 'n_estimators': [5, 10, 50, 100, 500],
                  'min_samples_split': [5, 10, 100, 500]}

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='gini', class_weight='balanced')
rf_grid = GridSearchCV(rf, param_grid=hyperparameter, cv=3, verbose=1, n_jobs=-1)
rf_grid_results_tr, rf_grid_results_test = model(rf_grid, X_train_3.toarray(), Y_train_new, X_test_3.toarray(),
                                                 Y_test_new)
print_grid_search_attributes(rf_grid)
rf_gs = rf_grid.best_estimator_
y_tr_pred = rf_gs.predict(X_train_3.toarray())
y_test_pred = rf_gs.predict(X_test_3.toarray())
rf_tpr_fpr_train = tpr_fpr_func_2(Y_train_new, y_tr_pred)
rf_tpr_fpr_test = tpr_fpr_func_2(Y_test_new, y_test_pred)
rf_tpr_fpr_train
rf_tpr_fpr_test
# https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python



from sklearn.metrics import roc_curve

clf = rf_grid.best_estimator_
clf.fit(X_train_3, Y_train_new)
Y_1 = clf.predict_proba(X_train_3)
Y1 = Y_1[:, 1]
train_fpr, train_tpr, tr_threshold = roc_curve(Y_train_new, Y1)
Y_2 = clf.predict_proba(X_test_3)
Y2 = Y_2[:, 1]
test_fpr, test_tpr, te_threshold = roc_curve(Y_test_new, Y2)
plt.plot(train_fpr, train_tpr, label='Train_AUC = ' + str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label='Test_AUC = ' + str(auc(test_fpr, test_tpr)))
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.title('AUC')
plt.legend()
plt.grid()
plt.show()
rf_grid_results_tr
rf_grid_results_test
joblib.dump(rf_grid.best_estimator_, 'rf_grid_3.pkl')


hyperparameter = {'max_depth': [2, 3, 5, 7, 10], 'n_estimators': [10, 50, 100, 200, 500]}

from xgboost import XGBClassifier

xgb = XGBClassifier(objective='binary:logistic', n_jobs=-1)
xgb_grid = RandomizedSearchCV(xgb, param_distributions=hyperparameter, cv=3, verbose=1, n_jobs=-1)
xgb_grid_results_tr, xgb_grid_results_test = model(xgb_grid, X_train_3.toarray(), Y_train_new,
                                                   X_test_3.toarray(), Y_test_new)
print_grid_search_attributes(xgb_grid)
xgb_gs = xgb_grid.best_estimator_
y_tr_pred = xgb_gs.predict(X_train_3.toarray())
y_test_pred = xgb_gs.predict(X_test_3.toarray())
xgb_tpr_fpr_train = tpr_fpr_func_2(Y_train_new, y_tr_pred)
xgb_tpr_fpr_test = tpr_fpr_func_2(Y_test_new, y_test_pred)
# https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python

from sklearn.metrics import roc_curve

clf = xgb_grid.best_estimator_
clf.fit(X_train_3, Y_train_new)
Y_1 = clf.predict_proba(X_train_3)
Y1 = Y_1[:, 1]
train_fpr, train_tpr, tr_threshold = roc_curve(Y_train_new, Y1)
Y_2 = clf.predict_proba(X_test_3)
Y2 = Y_2[:, 1]
test_fpr, test_tpr, te_threshold = roc_curve(Y_test_new, Y2)
plt.plot(train_fpr, train_tpr, label='Train_AUC = ' + str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label='Test_AUC = ' + str(auc(test_fpr, test_tpr)))
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.title('AUC')
plt.legend()
plt.grid()
plt.show()
xgb_grid_results_tr
xgb_grid_results_test
xgb_tpr_fpr_train
xgb_tpr_fpr_test
joblib.dump(xgb_grid.best_estimator_, 'xgb_grid_3.pkl')



from prettytable import PrettyTable

x = PrettyTable()

x.field_names = ['Model', 'Train_AUC', 'Train f1-score', 'Train TPR', 'Train FPR', 'Test AUC', 'Test f1-score',
                 'Test TPR', 'Test FPR']
x.add_row(['DT - 3', '0.9999', '0.9998', '99.98%', '0.00009%', '0.9991', '0.9989', '99.87%', '0.08%'])
x.add_row(['RF - 3', '0.9999', '0.9998', '99.99%', '0.013%', '0.9999', '0.9990', '99.94%', '0.13%'])
x.add_row(['XGB - 3', '1.0', '1.0', '100.0%', '0.0%', '0.9999', '0.9992', '99.94%', '0.10%'])

print(x)









