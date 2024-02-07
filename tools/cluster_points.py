from sklearn.cluster import DBSCAN

# define DBSCAN model. see ref https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
def cluster_points(points, eps=5, min_samples=30):
    '''cluster points with DBSCAN
    input:  numpy.ndarray (N,3)
    return: points (P, 3),  cluster_id (P, 1)
    '''
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    return points[db.labels_ != -1], db.labels_[db.labels_ != -1]