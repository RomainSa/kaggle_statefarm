import numpy as np
from joblib import Parallel, delayed

X_train = np.load('/home/ubuntu/data/kaggle_statefarm/train_features.npy')
X_test = np.load('/home/ubuntu/data/kaggle_statefarm/test_features.npy')
names = np.load('/home/ubuntu/data/kaggle_statefarm/test_names.npy')
y_train = np.load('/home/ubuntu/data/kaggle_statefarm/train_targets.npy')

def predict_proba(X):
  global knn
  return knn.predict_proba(X)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=100, algorithm='brute', metric='braycurtis')
knn.fit(X_train, y_train)
predictions_chunks = Parallel(n_jobs=30, backend="threading")(delayed(predict_proba)(X_test[i*1000:(i+1)*1000, :]) for i in range(80))
predictions = np.empty((0, 10))
for predictions_chunk in predictions_chunks:
  predictions = np.concatenate((predictions, predictions_chunk), axis=0)

print 'Saving predictions to csv...'
with open('submission_' + str(np.random.rand())[2:] + '.csv', 'w+') as f:
  f.write('img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n')
  for i, p in enumerate(predictions):
    f.write(names[i].split('/')[-1] + ',' + ','.join([str(x) for x in p]) + '\n')

