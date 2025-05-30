! pip install ucimlrepo
import pandas as pd
import numpy as np
from sklearn import datasets
import math
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from ucimlrepo import fetch_ucirepo

# fetch dataset
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

# metadata
print(heart_disease.metadata)

# variable information
print(heart_disease.variables)

#standandization
def standandization(target_table):
    f_table = target_table.copy()
    for col in f_table.columns[:-1]:
        f_table_mean = f_table[col].mean()
        f_table_std = f_table[col].std()
        for row in f_table.index:
            f_table.loc[row, col] = (f_table.loc[row, col] - f_table_mean) / f_table_std
    return f_table

# Preprocessing the data

vessel = heart_disease.data.features['ca']
vessel_mean = int(math.ceil(heart_disease.data.features['ca'].mean()))
thal = heart_disease.data.features['thal']
thal_mean = int(math.ceil(heart_disease.data.features['thal'].mean()))
vessel.fillna(vessel_mean, inplace=True)
thal.fillna(thal_mean, inplace=True)

standandized_table = standandization(heart_disease.data.original)

top_features = []
for feature in standandized_table.columns[:-1]:
  # pos_feature = standandized_table.loc[np.where(heart_disease.data.targets != 0)[0],[feature, 'num']]
  # neg_feature = standandized_table.loc[np.where(heart_disease.data.targets == 0)[0],[feature, 'num']]
  pos_feature = standandized_table.loc[np.where(heart_disease.data.targets != 0)[0],feature]
  neg_feature = standandized_table.loc[np.where(heart_disease.data.targets == 0)[0],feature]
  pos_feature_mean = pos_feature.mean()
  neg_feature_mean = neg_feature.mean()
  value = np.square(pos_feature_mean - neg_feature_mean)
  cat_type = heart_disease.variables.loc[heart_disease.variables.name == feature, 'type'].values[0]
  top_features.append([feature, value, cat_type])

top_features.sort(key=lambda x: x[1], reverse=True)
pd.DataFrame(top_features)

smd = pd.DataFrame(top_features, columns = ['name', 'smd', 'cat_type'])
smd.sort_values(by = 'smd')

smd_categorical = smd.loc[smd.cat_type == 'Categorical','smd'].to_numpy()  # SMD for 3 categorical features
smd_continuous = smd.loc[smd.cat_type == 'Integer','smd'].to_numpy()      # SMD for 2 continuous features

smd_categorical_total = np.sum(smd_categorical)
smd_continuous_total = np.sum(smd_continuous)

w_categorical = smd_categorical_total / (smd_categorical_total + smd_continuous_total)
w_continuous = smd_continuous_total / (smd_categorical_total + smd_continuous_total)

cat_ix = np.where((heart_disease.variables.type == "Categorical").astype(int)[:-1])[0]
cont_ix = np.where((heart_disease.variables.type == "Integer").astype(int)[:-1])[0]

euclidean = lambda x1, x2: np.sqrt(np.sum((x1 - x2)**2, axis=-1))
manhattan = lambda x1, x2: np.sum(np.abs(x1 - x2), axis=-1)
hamming = lambda x1, x2: np.sum((x1 != x2).astype(int), axis=-1)


class KNN:

    def __init__(self, K=1, cont_dist_fn=euclidean, cat_dist_fn=hamming):
        self.cont_dist_fn = cont_dist_fn
        self.cat_dist_fn = cat_dist_fn
        self.K = K
        self.w_categorical = None
        self.w_continuous = None
        self.cat_ix = None
        self.cont_ix = None
        return

    def fit(self, x, y, cat_type):
        ''' Store the training data using this method as it is a lazy learner'''
        self.x = x
        self.y = y
        self.cat_type = cat_type
        self.C = np.max(y) + 1 # for us first row will always be categorical or continuous type information
        return self

    def manage_categorical_continuous_input(self):
        '''Computes feature importance of either categorical and continuous input features.
        Computes their contribution weights when distances for either feature type.
        Summed together after appropriate distance functions are used.
        Different methods can be used'''
        top_features = []
        for feature in self.x.columns:
            pos_feature_mean = self.x.loc[np.where(self.y != 0)[0],feature].mean()
            neg_feature_mean = self.x.loc[np.where(self.y == 0)[0],feature].mean()
            smd = np.square(pos_feature_mean - neg_feature_mean) # square mean difference
            cat = self.cat_type.iloc[0,np.where(self.x.columns == feature)[0]].values[0]
            top_features.append([feature, smd, cat])

        smd = pd.DataFrame(top_features, columns = ['name', 'smd', 'cat_type'])
        smd.sort_values(by = 'smd')
        smd_categorical_total = smd.loc[smd.cat_type == 'Categorical','smd'].to_numpy().sum()  # SMD for 3 categorical features
        smd_continuous_total = smd.loc[smd.cat_type == 'Integer','smd'].to_numpy()  .sum()    # SMD for 2 continuous features

        self.w_categorical = smd_categorical_total / (smd_categorical_total + smd_continuous_total)
        self.w_continuous = smd_continuous_total / (smd_categorical_total + smd_continuous_total)

        self.cat_ix = np.where(self.cat_type.values.flatten() == 'Categorical')[0]
        self.cont_ix = np.where(self.cat_type.values.flatten() == 'Integer')[0]

        return self.w_categorical, self.w_continuous, self.cat_ix, self.cont_ix

    def predict(self, x_test):
        ''' Makes a prediction using the stored training data and the test data given as argument'''
        num_test = x_test.shape[0]

        # if the input shapes are [1,N1,F] and [N2,1,F] then output shape is [N2,N1]
        # calculate distance between the training & test samples and returns an array of shape [num_test, num_train]
        # self.x is in shape (100, 2), x_test is in shape (50, 2)
        # self.x[None, :, :] is in shape (1, 100, 2), and x_test[:,None,:] is in shape (50, 1, 2)
        ### x_test is 50 layers of 1 row, 2 columns, and will broadcast over each row of self.x
        # result: (x_test.shape[0], self.x.shape[0])

        # 1x264x13
        # 61x1x13

        # standardize continuous features only
        # squared mean differences
        # should i standardize my values before performing squared mean differences??????

        self.w_categorical, self.w_continuous, self.cat_ix, self.cont_ix = self.manage_categorical_continuous_input()

        if len(set(self.cat_type.values.flatten())) > 1:
            self.x = self.x.astype(float)
            for col in self.cont_ix:
                col_mean = self.x.iloc[:, col].mean()
                col_std = self.x.iloc[:, col].std()
                self.x.iloc[:, col] = (self.x.iloc[:, col].astype(np.float64) - col_mean) / col_std

            self.x = self.x.to_numpy()
            x_test = x_test.to_numpy()
            cat_distances = self.cat_dist_fn(self.x[None,:,self.cat_ix], x_test[:,None,self.cat_ix]) # 61x264
            cont_distances = self.cont_dist_fn(self.x[None,:,self.cont_ix], x_test[:,None,self.cont_ix]) #61x264

            distances = self.w_categorical*(cat_distances) + self.w_continuous*(cont_distances)
        else:
            distances = self.cont_dist_fn(self.x[None,:,:], x_test[:,None,:]) # distance between one training example and every test point, one distance value per

        # print(distances)
        #ith-row of knns stores the indices of k closest training samples to the ith-test sample
        knns = np.zeros((num_test, self.K), dtype=int)
        #ith-row of y_prob has the probability distribution over C classes
        y_prob = np.zeros((num_test, self.C))
        self.y = self.y.to_numpy().flatten()
        for i in range(num_test):
            # print(i)
            # print(distances[i].shape)
            # print(distances[i])

            knns[i,:] = np.argsort(distances[i])[:self.K] # sorts distances in ascending order (smallest to largest), selects only first
            # print(knns[i,:])
            # print(len(self.y))
            # print(self.y[knns[i,:]].flatten())

            y_prob[i,:] = np.bincount(self.y[knns[i,:]], minlength=self.C) #counts the number of instances of each class in the K-closest training samples, minimum amount of bins
        #y_prob /= np.sum(y_prob, axis=-1, keepdims=True)
        #simply divide by K to get a probability distribution
        y_prob /= self.K
        y_pred = np.argmax(y_prob, axis = -1)

        return y_prob, knns, y_pred
    
# functions
def train_test_split(features, labels, split):

    ix = np.arange(features.shape[0])
    np.random.shuffle(ix)

    X_train, X_test = features[ix[:int(len(ix)*split)],:], features[ix[int(len(ix)*split):],:]
    y_train, y_test = labels[ix[:int(len(ix)*split)]], labels[ix[int(len(ix)*split):]]

    X_train = pd.DataFrame(X_train, columns=heart_disease.data.features.columns)
    X_test = pd.DataFrame(X_test, columns=heart_disease.data.features.columns)
    y_train = pd.DataFrame(y_train, columns=heart_disease.data.targets.columns)
    y_test = pd.DataFrame(y_test, columns=heart_disease.data.targets.columns)

    X_train.reset_index(inplace = True, drop = True)
    X_test.reset_index(inplace = True, drop = True)
    y_train.reset_index(inplace = True, drop = True)
    y_test.reset_index(inplace = True, drop = True)

    return X_train, X_test, y_train, y_test

def evaluate_acc(y_pred, y_test, verbose):
    classification_accuracy = np.sum(y_pred == y_test)/y_test.shape[0]
    if verbose:
        print(f'classification accuracy is {classification_accuracy*100:.1f}%')
    else:
        return classification_accuracy

# further split the training data into 50% training and 50% validation
x_train_tr, y_train_tr = x_train[:50], y_train[:50]
x_train_va, y_train_va = x_train[50:], y_train[50:]

model_choices=[]
valid_acc = []

n_valid = y_train_va.shape[0]

for k in range(1,20):
    knn = KNN(K=k) # create a KNN object (OOP)
    # y_train_va_prob,_ = knn.fit(x_train, y_train).predict(x_train_va) # wrong
    y_train_va_prob,_knns, y_train_va_pred = knn.fit(x_train_tr, y_train_tr).predict(x_train_va, cat_ix, cont_ix, w_categorical, w_continuous, combined = True) # bug fixed
    accuracy = np.sum(y_train_va_pred == y_train_va)/n_valid
    model_choices.append(k)
    valid_acc.append(accuracy)

# use the best K to predict test data
best_valid_K = model_choices[valid_acc.index(max(valid_acc))]
knn = KNN(K=best_valid_K)
y_test_prob, knns, y_test_pred = knn.fit(x_train, y_train).predict(x_test, cat_ix, cont_ix, w_categorical, w_continuous, combined = True)
test_accuracy = np.sum(y_test_pred == y_test)/y_test.shape[0]
print(f'best K = {best_valid_K}, test accuracy = {test_accuracy}')

plt.plot(model_choices, valid_acc, marker='o', color='blue', label='validation')
plt.plot(best_valid_K, test_accuracy, marker='*', color='red', label='testing')
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.legend(loc='best')
plt.savefig('iris_KNN_chooseK.png',dpi=300,bbox_inches='tight')