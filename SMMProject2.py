import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sklearn
from sklearn import tree
from sklearn import ensemble
from sklearn import neighbors
from sklearn import linear_model
from sklearn import metrics


def get_file_paths():
    user_files = []
    user_files.append('C:/Users/Simran Singh/Desktop/cresci-2017.csv/datasets_full.csv/genuine_accounts.csv/genuine_accounts.csv/users.csv')

    social_bot_files = []
    social_bot_files.append('C:/Users/Simran Singh/Desktop/cresci-2017.csv/datasets_full.csv/social_spambots_1.csv/social_spambots_1.csv/users.csv')
    social_bot_files.append('C:/Users/Simran Singh/Desktop/cresci-2017.csv/datasets_full.csv/social_spambots_2.csv/social_spambots_2.csv/users.csv')
    social_bot_files.append('C:/Users/Simran Singh/Desktop/cresci-2017.csv/datasets_full.csv/social_spambots_3.csv/social_spambots_3.csv/users.csv')

    spam_bot_files = []
    spam_bot_files.append('C:/Users/Simran Singh/Desktop/cresci-2017.csv/datasets_full.csv/traditional_spambots_1.csv/traditional_spambots_1.csv/users.csv')
    spam_bot_files.append('C:/Users/Simran Singh/Desktop/cresci-2017.csv/datasets_full.csv/traditional_spambots_2.csv/traditional_spambots_2.csv/users.csv')
    spam_bot_files.append('C:/Users/Simran Singh/Desktop/cresci-2017.csv/datasets_full.csv/traditional_spambots_3.csv/traditional_spambots_3.csv/users.csv')
    spam_bot_files.append('C:/Users/Simran Singh/Desktop/cresci-2017.csv/datasets_full.csv/traditional_spambots_4.csv/traditional_spambots_4.csv/users.csv')

    return user_files, social_bot_files, spam_bot_files


def pre_process(user_files, social_bot_files, spam_bot_files):
    final_data = pd.DataFrame({'A': []})

    for user_file in user_files:
        temp_data = pd.read_csv(user_file, encoding='latin1')
        temp_data = temp_data[
            ['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'lang',
             'time_zone', 'default_profile', 'geo_enabled', 'profile_use_background_image', 'profile_text_color',
             'profile_sidebar_border_color', 'profile_background_tile', 'profile_sidebar_fill_color',
             'profile_background_color', 'profile_link_color', 'utc_offset']]

        temp_data['label'] = int(0)

        final_data = final_data.append(temp_data, sort=True)

    final_data = final_data.drop('A', axis=1)

    for social_bot_file in social_bot_files:
        temp_data = pd.read_csv(social_bot_file, encoding='latin1')
        temp_data = temp_data[
            ['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'lang',
             'time_zone', 'default_profile', 'geo_enabled', 'profile_use_background_image', 'profile_text_color',
             'profile_sidebar_border_color', 'profile_background_tile', 'profile_sidebar_fill_color',
             'profile_background_color', 'profile_link_color', 'utc_offset']]

        temp_data['label'] = int(1)

        final_data = final_data.append(temp_data, sort=True)

    for spam_bot_file in spam_bot_files:
        temp_data = pd.read_csv(spam_bot_file, encoding='latin1')
        temp_data = temp_data[
            ['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'lang',
             'time_zone', 'default_profile', 'geo_enabled', 'profile_use_background_image', 'profile_text_color',
             'profile_sidebar_border_color', 'profile_background_tile', 'profile_sidebar_fill_color',
             'profile_background_color', 'profile_link_color', 'utc_offset']]

        temp_data['label'] = int(2)

        final_data = final_data.append(temp_data, sort=True)

    return final_data


def dictionary_encodings(column, feature):
    encoded_column = []
    encoding_dict = {}
    index = 0

    for i, row in column.iterrows():
        if row[feature] not in encoding_dict:
            encoding_dict[row[feature]] = index
            index = index + 1

    for i, row in column.iterrows():
        encoded_column.append(encoding_dict[row[feature]])

    return np.array(encoded_column)


def post_process(data, dictionary_encoding_features):
    data = data.fillna(0)

    for feature in dictionary_encoding_features:
        column = dictionary_encodings(data[:][[feature]], feature)
        data[feature] = column

    return data


def generate_data():
    user_files, social_bot_files, spam_bot_files = get_file_paths()
    data = pre_process(user_files, spam_bot_files, spam_bot_files)
    data = post_process(data, ['lang', 'time_zone', 'profile_text_color', 'profile_sidebar_border_color',
                               'profile_sidebar_fill_color', 'profile_background_color', 'profile_link_color'])

    y, X = data['label'], data.drop('label', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    return X_train.values, y_train.values, X_test.values, y_test.values


def main():
    X_train, y_train, X_test, y_test = generate_data()

    print("SVM")
    clf = sklearn.svm.LinearSVC(multi_class='ovr')
    clf = sklearn.svm.SVC(C=10, gamma=10, degree=8)
    clf.fit(X_train, y_train)
    y_pred_temp = clf.predict(X_test)
    y_pred_svm = np.expand_dims(y_pred_temp, 1)

    # for i in range(len(y_test.shape[0])):
    #     print(y_test[i])
    #
    # for i in range(len(y_pred_temp.shape[0])):
    #     print(y_pred_temp[i])

    correct_svm = 0
    for i in range(y_pred_svm.shape[0]):
        if y_pred_svm[i] == y_test[i]:
            correct_svm = correct_svm + 1
    print("Accuracy: ",correct_svm/y_pred_svm.shape[0])
    print("Confusion Matrix: ", metrics.confusion_matrix(y_test, y_pred_svm))
    print("Precision Score: ", metrics.precision_score(y_test, y_pred_svm, labels=[0, 1, 2], average='weighted'))
    print("Recall Score: ", metrics.recall_score(y_test, y_pred_svm, labels=[0, 1, 2], average='weighted'))
    print("F1 Score: ", metrics.f1_score(y_test, y_pred_svm, average='macro'))
    #print("AUC ROC Score: ",metrics.roc_auc_score(y_test, y_pred_svm, average='weighted'))


    print("Decision Tree")
    clf2 = sklearn.tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=19, min_samples_split=18)
    clf2.fit(X_train, y_train)
    y_pred_dt = clf2.predict(X_test)
    # print(y_pred_dt.shape)
    # for i in range(y_pred_dt.shape[0]):
    #     print(y_pred_dt[i])
    # print(y_test.shape)

    correct_dt = 0
    for i in range(y_pred_dt.shape[0]):
        if y_pred_dt[i] == y_test[i]:
            correct_dt = correct_dt + 1
    print("Accuracy: ",correct_dt/y_pred_dt.shape[0])
    print("Confusion Matrix: ", metrics.confusion_matrix(y_test, y_pred_dt))
    print("Precision Score: ", metrics.precision_score(y_test, y_pred_dt, labels=[0, 1, 2], average='weighted'))
    print("Recall Score: ", metrics.recall_score(y_test, y_pred_dt, labels=[0, 1, 2], average='weighted'))
    print("F1 Score: ", metrics.f1_score(y_test, y_pred_dt, average='macro'))
    # print("AUC ROC Score: ", metrics.roc_auc_score(y_test, y_pred_dt))


    print("Random Forest Classifier")
    clf3 = sklearn.ensemble.RandomForestClassifier(criterion='entropy', min_samples_leaf=19, min_samples_split=18)
    clf3.fit(X_train, y_train)
    y_pred_rf = clf3.predict(X_test)

    correct_rf = 0
    for i in range(y_pred_rf.shape[0]):
        if y_pred_rf[i] == y_train[i]:
            correct_rf = correct_rf + 1
    print("Accuracy: ",correct_rf/y_pred_rf.shape[0])
    print("Confusion Matrix: ", metrics.confusion_matrix(y_test, y_pred_rf))
    print("Precision Score: ", metrics.precision_score(y_test, y_pred_rf, labels=[0, 1, 2], average='weighted'))
    print("Recall Score: ", metrics.recall_score(y_test, y_pred_rf, labels=[0, 1, 2], average='weighted'))
    print("F1 Score: ", metrics.f1_score(y_test, y_pred_rf, average='macro'))
    # print("AUC ROC Score: ", metrics.roc_auc_score(y_test, y_pred_rf))

    print("K Nearest Neighbour")
    clf4 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=30, algorithm='kd_tree')
    clf4.fit(X_train, y_train)
    y_pred_kn = clf4.predict(X_test)

    correct_kn = 0
    for i in range(y_pred_kn.shape[0]):
        if y_pred_kn[i] == y_test[i]:
            correct_kn = correct_kn + 1
    print("Accuracy using K Nearest Neighbours: ",correct_kn/y_pred_kn.shape[0])
    print("Confusion Matrix: ", metrics.confusion_matrix(y_test, y_pred_kn))
    print("Precision Score: ", metrics.precision_score(y_test, y_pred_kn, labels=[0, 1, 2], average='weighted'))
    print("Recall Score: ", metrics.recall_score(y_test, y_pred_kn, labels=[0, 1, 2], average='weighted'))
    print("F1 Score: ", metrics.f1_score(y_test, y_pred_kn, average='macro'))
    # #print("AUC ROC Score: ", metrics.roc_auc_score(y_test, y_pred_kn))

    print("Adaboost Classifer")
    clf5 = sklearn.ensemble.AdaBoostClassifier(n_estimators=600, learning_rate=2.5, algorithm="SAMME")
    clf5.fit(X_train, y_train)
    y_pred_ada = clf5.predict(X_test)

    correct_ada = 0
    for i in range(y_pred_ada.shape[0]):
        if y_pred_ada[i] == y_test[i]:
            correct_ada = correct_ada + 1
    print("Accuracy using Adaboost Classifier: ", correct_ada / y_pred_ada.shape[0])
    print("Confusion Matrix: ", metrics.confusion_matrix(y_test, y_pred_ada))
    print("Precision Score: ", metrics.precision_score(y_test, y_pred_ada, labels=[0, 1, 2], average='weighted'))
    print("Recall Score: ", metrics.recall_score(y_test, y_pred_ada, labels=[0, 1, 2], average='weighted'))
    print("F1 Score: ", metrics.f1_score(y_test, y_pred_ada, average='macro'))
    #print("AUC ROC Score: ", metrics.roc_auc_score(y_test, y_pred_ada))


    print("Logistic Regression")
    clf6 = sklearn.linear_model.LogisticRegression(multi_class='multinomial',solver='newton-cg',max_iter=2000)
    clf6.fit(X_train, y_train)
    y_pred_lr = clf6.predict(X_test)

    correct_lr = 0
    for i in range(y_pred_lr.shape[0]):
        if y_pred_lr[i] == y_test[i]:
            correct_lr = correct_lr+1
    print("Accuracy using Linear Regression as the classifier: ",correct_lr/y_pred_lr.shape[0])
    print("Confusion Matrix: ", metrics.confusion_matrix(y_test,y_pred_lr))
    #print("Accuracy using Linear Regression as the classifier: ",metrics.accuracy_score(y_test, y_pred_lr))
    print("Precision Score: ",metrics.precision_score(y_test, y_pred_lr, labels=[0,1,2] ,average='weighted'))
    print("Recall Score: ",metrics.recall_score(y_test, y_pred_lr, labels=[0,1,2], average='weighted'))
    print("F1 Score: ",metrics.f1_score(y_test, y_pred_lr, average='macro'))
    #print("AUC ROC Score: ", metrics.roc_auc_score(y_test, y_pred_lr))





main()