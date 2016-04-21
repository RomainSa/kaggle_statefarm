import os
import numpy as np
from sklearn.cluster import KMeans

features = np.load('/home/ubuntu/data/kaggle_statefarm/test_features.npy')
names = np.load('/home/ubuntu/data/kaggle_statefarm/test_names.npy')
n_drivers = 55

# make cluster predictions or load them if needed
if not os.path.isfile('predictions.npy'):
    km = KMeans(n_clusters=n_drivers, init='k-means++', n_init=10, max_iter=3000, tol=0.0001,
                precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=-1)
    km.fit(features)
    predictions = km.predict(features)
    np.save('predictions.npy', predictions)

cluster_predictions = np.load('predictions.npy')

# loads CNN predictions
category_predictions = np.load('mnist_predictions.npy')
category_predictions_names = np.load('mnist_predictions_names.npy')

final_predictions = np.empty((0, 10))
final_names = np.empty((0, ))
for i in range(n_drivers):
    # get cluster data
    cluster_features = features[cluster_predictions == i]
    cluster_names = names[cluster_predictions == i]
    # perform a new clustering
    cluster_km = KMeans(n_clusters=10, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=-1)
    cluster_km.fit(cluster_features)
    subcluster_predictions = cluster_km.predict(cluster_features)
    for j in range(10):
        subcategory_predictions = np.empty((0, 10))
        subcategory_names = cluster_names[subcluster_predictions == j]
        for name in subcategory_names:
            name = \
            name.replace('//c0', '').replace('//c1', '').replace('//c2', '').replace('//c3', '').replace('//c4', '')\
                .replace('//c5', '').replace('//c6', '').replace('//c7', '').replace('//c8', '').replace('//c9', '')
            data = category_predictions[category_predictions_names == name]
            subcategory_predictions = np.concatenate((subcategory_predictions, data), axis=0)
        subcategory_predictions_mean = subcategory_predictions.mean(axis=0)
        final_predictions = np.concatenate((final_predictions,
                                            np.repeat(subcategory_predictions_mean, subcategory_names.shape[0]).reshape((10, subcategory_names.shape[0])).T))
        final_names = np.concatenate((final_names, subcategory_names))


with open('submission_' + str(np.random.rand())[2:] + '.csv', 'w+') as f:
    f.write('img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n')
    for i, p in enumerate(final_predictions):
        f.write(final_names[i].split('/')[-1] + ',' + ','.join([str(x) for x in p]) + '\n')

# TODO faire en sorte que quand une classe a été choisie, les autres sont a 0
