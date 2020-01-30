import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from featureExtraction import feature_extraction
import joblib

def concatenate(data):
    n = data.shape[0]
    space = np.repeat(np.array([' ']), n).reshape(n, 1)
    data['sapce'] = space
    data['con'] = data['turn1']  + data['sapce'] + data['turn3']
    return data

#  Read count features
count_feature = pd.read_csv("data/features.csv")
test_count_feature = pd.read_csv("data/dev_features.csv")
word_count = pd.read_csv("data/train_count.csv")
test_word_count = pd.read_csv("data/test_count.csv")

# Read cleaned training data
data_clean = pd.read_csv("data/preprocessed_train_delete_stopwords_cleanText.csv")
data_clean_withemo = pd.read_csv("data/preprocessed_train_data_Emoticon.csv")
data_clean.fillna('', inplace=True)
data_clean_withemo.fillna('', inplace=True)
data = concatenate(data_clean)
data_withemo = concatenate(data_clean_withemo)
data_withemo.rename(columns = {'con':'conemo'}, inplace = True)
data_all = pd.concat([data, data_withemo['conemo'], count_feature['turn13_symbol'], count_feature['turn13_upper'], count_feature['turn13_marks'],word_count['turn13_word']], axis=1)


# all training data
train_X = data_all['con']
train_X_emo = data_all['conemo']

# split training set to train and test sets
tr_tr_data,tr_tst_data = train_test_split(data_all, test_size=0.3)
tr_tr_X = tr_tr_data['con']
tr_tr_X_emo = tr_tr_data['conemo']
tr_tst_X = tr_tst_data['con']
tr_tst_X_emo = tr_tst_data['conemo']

#  Save train_train and train_test labels to files
tr_tr_data[['id','label']].to_csv("data/train_train_labels1.csv")
tr_tst_data[['id','label']].to_csv("data/train_test_labels1.csv")

#  Read cleaned test data
test_clean = pd.read_csv("data/preprocessed_test_delete_stopwords_cleanText.csv")
test_clean_withemo = pd.read_csv("data/preprocessed_test_data_emoticon.csv")
test_clean.fillna('', inplace=True)
test_clean_withemo.fillna('', inplace=True)
test_data = concatenate(test_clean)
tets_data_withemo = concatenate(test_clean_withemo)
tets_data_withemo.rename(columns = {'con':'conemo'}, inplace = True)
test_all =  pd.concat([test_data, tets_data_withemo['conemo'], test_count_feature['turn13_symbol'], test_count_feature['turn13_upper'], test_count_feature['turn13_marks'], test_word_count['turn13_word']], axis=1)
print(test_all.head())
test_X = test_all['con']
test_X_withemo = test_all['conemo']



# Feature extraction for train_train and train_test
features = feature_extraction(tr_tr_X, tr_tr_X_emo)
train_train_vecs = features.get_features()

train_train_vecs.append(np.array(tr_tr_data['turn13_symbol']).reshape(-1, 1))
train_train_vecs.append(np.array(tr_tr_data['turn13_upper']).reshape(-1, 1))
train_train_vecs.append(np.array(tr_tr_data['turn13_marks']).reshape(-1, 1))
train_train_vecs.append(np.array(tr_tr_data['turn13_word']).reshape(-1, 1))

train_test_vecs = features.get_features(tr_tst_X, tr_tst_X_emo)
train_test_vecs.append(np.array(tr_tst_data['turn13_symbol']).reshape(-1, 1))
train_test_vecs.append(np.array(tr_tst_data['turn13_upper']).reshape(-1, 1))
train_test_vecs.append(np.array(tr_tst_data['turn13_marks']).reshape(-1, 1))
train_test_vecs.append(np.array(tr_tst_data['turn13_word']).reshape(-1,1))


# Feature extraction for train and test
features = feature_extraction(train_X, train_X_emo)
train_vecs = features.get_features()
train_vecs.append(np.array(data_all['turn13_symbol']).reshape(-1, 1))
train_vecs.append(np.array(data_all['turn13_upper']).reshape(-1, 1))
train_vecs.append(np.array(data_all['turn13_marks']).reshape(-1, 1))
train_vecs.append(np.array(data_all['turn13_word']).reshape(-1, 1))
test_vecs = features.get_features(test_X, test_X_withemo)
test_vecs.append(np.array(test_all['turn13_symbol']).reshape(-1, 1))
test_vecs.append(np.array(test_all['turn13_upper']).reshape(-1, 1))
test_vecs.append(np.array(test_all['turn13_marks']).reshape(-1, 1))
test_vecs.append(np.array(test_all['turn13_word']).reshape(-1, 1))



#  dump feature vectors to files
joblib.dump(train_train_vecs, 'data/train_train_vecs.pkl')
joblib.dump(train_test_vecs, 'data/train_test_vecs.pkl')
joblib.dump(train_vecs, 'data/train_vecs.pkl')
joblib.dump(test_vecs, 'data/test_vecs.pkl')



