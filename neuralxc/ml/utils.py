from ..formatter import expand, atomic_shape, system_shape
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator


def find_attr_in_tree(file, tree, attr):

    if attr in file[tree].attrs:
        return file[tree].attrs[attr]

    tree_list = tree.split('/')

    for i in range(1, len(tree_list)):
        subtree = '/'.join(tree_list[:-i])
        if attr in file[subtree].attrs:
            return file[subtree].attrs[attr]


def load_data(datafile, baseline, reference, basis_key, percentile_cutoff=0.0,
                E0 = None):

    data_base = datafile[baseline +'/energy']
    data_ref = datafile[reference +'/energy']

    if E0 == None:
        E0_base = find_attr_in_tree(datafile, baseline, 'E0')
        E0_ref = find_attr_in_tree(datafile, reference, 'E0')
    else:
        E0_base = E0
        E0_ref = 0
    if E0_base == None:
        print('Warning: E0 for baseline data not found, setting to 0')
        E0_base = 0
    if E0_ref == None:
        print('Warning: E0 for reference data not found, setting to 0')
        E0_ref = 0

    print('E0 base', E0_base)
    print('E0 ref', E0_ref)
    tar = (data_ref[:] - E0_ref) - (data_base[:] -  E0_base)
    tar = tar.real
    # if percentile_cutoff > 0:
    #     lim1 = np.percentile(tar, percentile_cutoff*100)
    #     lim2 = np.percentile(tar, (1 - percentile_cutoff)*100)
    #     min_lim, max_lim = min(lim1,lim2), max(lim1,lim2)
    #     filter2 = (tar > min_lim) & (tar < max_lim)
    # else:
    #     filter2 = [True]*len(tar)
    #
    # filter = filter2
    data_base = datafile[baseline +'/density/' + basis_key]
    data_base = data_base[:, :]

    # feat = {}
    # all_species = find_attr_in_tree(datafile, baseline, 'species')
    # all_species = [c for c in all_species]
    # unique_species = np.unique(all_species)
    # attrs = {}

    # for spec in unique_species:
        # attrs.update({spec: {'_'.join(attr.split('_')[:-1]): find_attr_in_tree(datafile, baseline, attr)\
             # for attr in ['n_' + spec,'l_' + spec, 'r_o_' + spec]}})

# if grouped:
#     for spec in unique_species:
#         feat[spec] = []
#
#     current_loc = 0
#     for spec in all_species:
#         len_descr = attrs[spec]['n'] * sum([2 * l + 1 for l in range(attrs[spec]['l'])])
#         feat[spec].append(data_base[:, current_loc:current_loc + len_descr])
#         current_loc += len_descr
#
#     for spec in unique_species:
#         feat[spec] = np.array(feat[spec]).swapaxes(0, 1)
#
#     return feat, tar, attrs, ''.join(all_species)
# else:
    return data_base, tar


class SampleSelector(BaseEstimator):
    def __init__(self, n_instances, random_state=None):
        self._n_instances = n_instances
        self._random_state = random_state

    def fit(self, X, y=None):
        pass

    def predict(self, X):

        if isinstance(X, tuple):
            X = X[0]
        if not isinstance(X, list):
            X = [X]
        picks = [[] * len(X)]
        for idx, key, data in expand(X):
            data = data[0]
            indices = np.zeros(data.shape[:-1], dtype=int)
            indices[:] = np.arange(len(data)).reshape(-1, 1)
            picks[idx] += self.sample_clusters(atomic_shape(data),
                                               indices.flatten(),
                                               self._n_instances,
                                               picked=picks[idx],
                                               random_state=self._random_state)
        return picks[0][:self._n_instances]

    @staticmethod
    def sample_clusters(data,
                        indices,
                        n_samples,
                        picked=[],
                        cluster_method=KMeans,
                        pca_threshold=0.999,
                        random_state=None):

        pca = PCA(n_components=pca_threshold, svd_solver='full')
        pca_results = pca.fit_transform(data)

        clustering = cluster_method(n_samples, random_state=random_state).fit(pca_results)
        labels = clustering.labels_

        sampled = []

        np.random.seed(random_state)
        for label in np.unique(labels):  # Loop over clusters
            n = sum(labels == label)
            contained = False
            for i in indices[labels == label]:
                if i in picked:
                    contained = True  # Was a representative of this cluster already picked?
                    break
            if not contained:
                choice = np.random.choice(range(n), size=1, replace=False)
                sampled.append(indices[labels == label][choice[0]])

        return sampled
