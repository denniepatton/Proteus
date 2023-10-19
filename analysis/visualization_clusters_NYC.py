
import umap
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import umap.plot as ump
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
# tf.compat.v1.enable_eager_execution() 


def load_data(path, additional_path= None):
    
    data = np.load(path)
    print("loaded data")
    ids = data["ids"]
    samples = np.array([i.split("_")[0].strip() for i in ids])
    scalars = data["scalars"]
    labels = data["labels"]

    if additional_path is not None:
        data2 = np.load(additional_path)
        print("loaded additional data")
        ids2 = data2["ids"]
        samples2 = [i.split("_")[0].strip() for i in ids2]
        scalars2 = data2["scalars"]
        labels2 = data2["labels"]
        ids = np.concatenate([ids, ids2])
        samples = np.concatenate([samples, samples2])
        scalars = np.concatenate([scalars, scalars2])
        labels = np.concatenate([labels, labels2])
    print(ids.shape, samples.shape, scalars.shape, labels.shape)
    return ids, samples, scalars, labels

def normalize(array):
    return StandardScaler().fit_transform(array)

def fit_umap(array):
    # create UMAP dimensionality-reduction model
    reducer = umap.UMAP()
    # fit to scalar data
    embedding = reducer.fit_transform(array)
    print("reduced by umap")
    return reducer, embedding

def plot_embedding(embedding, categories, output):
    unique_colors = list(set(categories))
    sample_labels = [unique_colors.index(s) for s in categories]
    scatter_artists = {}
    for sample in unique_colors:
        indices = np.where(np.array(categories) == sample)[0]
        scatter_artists[sample] = plt.scatter(
            embedding[indices, 0],
            embedding[indices, 1],
            s=5,
            alpha=0.3,
            label=sample
        )
    scatter_list = [scatter_artists[sample] for sample in unique_colors]
    plt.legend(handles=scatter_list, title='Variable')
    plt.show()
    plt.savefig(output)
    plt.clf()

def plot_umap(umap, labels, output, filters = None):
    plt.clf()
    fig, ax = plt.subplots()
    if filters is None:
        ump.points(umap, labels = labels, ax = ax, alpha = 0.8)
    else:
        ump.points(umap, labels = labels, ax = ax, alpha = 0.8, subset_points = filters)
    plt.show()
    fig.savefig(output)


def main():
    training_path = '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/training.npz'
    testing_path = '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/testing.npz'
    ids, samples, scalars, labels = load_data(training_path, testing_path)
    scalars_norm = normalize(scalars)
    umap, embedding = fit_umap(scalars_norm)
    plot_umap(umap, samples, '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/analysis/plots/umaps/umap_by_samples.pdf')
    # plot_embedding(embedding, samples, '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/analysis/plots/umap_by_samples.png')
    label_map = {
        0: "Downregulated",
        1: "Neutral", 
        2: "Upregulated"
    }
    labels = labels.tolist()
    labels = np.array([label_map[val] for val in labels])
    plot_umap(umap, labels, '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/analysis/plots/umaps/umap_by_labels.pdf')
    low_coverage = ["189-3", "174-1", "81"]
    high_coverage = ["173-1", "176CR", "93"]
    plot_umap(umap, samples, '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/analysis/plots/umaps/umap_by_samples_low_coverage.pdf', \
        filters = np.isin(samples, low_coverage))
    plot_umap(umap, samples, '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/analysis/plots/umaps/umap_by_samples_high_coverage.pdf', \
        filters = np.isin(samples, high_coverage))
    print("done")
if __name__ == "__main__":
    main()

# unique_samples = ["Downregulated", "Neutral", "Upregulated"]
# sample_labels = [unique_samples.index(s) for s in labels]
# scatter_artists = {}
# for i, sample in enumerate(unique_samples):
#     indices = np.where(np.array(samples) == sample)[0]
#     scatter_artists[sample] = plt.scatter(
#         embedding[indices, 0],
#         embedding[indices, 1],
#         s=3,
#         alpha=0.7,
#         label=sample
#     )


# scatter_list = [scatter_artists[sample] for sample in unique_samples]
# plt.legend(handles=scatter_list, title='Samples')
# plt.show()
# plt.savefig('/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/analysis/umap_labels_test.pdf')
# plt.clf()

# # plot
# scatter = plt.scatter(
#     embedding[:, 0],
#     embedding[:, 1],
#     c=labels,
#     s=2)
# # Create a legend
# legend = plt.legend(
#     *scatter.legend_elements(),
#     title='Labels',
#     labels=["Downregulated", "Neutral", "Upregulated"]
# )
# plt.title("UMAP, colored by labels")
# plt.show()
# plt.savefig('/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/analysis/umap_labels_test.pdf')
# plt.clf()




# # path to X, y
# scalar_data ='/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/scalars_train_tnsr.pkl' 
# label_data = '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/labels_train_tnsr.pkl'



# # open tensors, convert to np arrays
# with open(scalar_data, 'rb') as f:
#     scalar_tensor: tf.Tensor = pickle.load(f)
#     scalar_np = scalar_tensor.numpy()
# with open(label_data, 'rb') as f:
#     label_tensor: tf.Tensor = pickle.load(f)
#     label_np = label_tensor.numpy()
#     # Define the mapping dictionary
#     mapping = {0: 'Downregulated', 1: 'Unchanged', 2: 'Upregulated'}

#     # Map the values using the mapping dictionary
#     labels = np.array([mapping[val] for val in label_np])
# print(f"Scalar shape {scalar_tensor.shape}")
# print(f"Label shape {label_tensor.shape}")
# # filter NA values and normalize arrays
# nan_rows = np.isnan(scalar_np).any(axis=1)
# scalar_np = scalar_np[~nan_rows]
# label_np = label_np[~nan_rows]
# scalar_norm = StandardScaler().fit_transform(scalar_np)
# print("normalized array")

# import_umap = '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/analysis/umap_model.pkl'

# create UMAP dimensionality-reduction model
# reducer = umap.UMAP()
# # fit to scalar data
# embedding = reducer.fit_transform(scalar_norm)
# print("reduced by umap")


# # plot
# scatter = plt.scatter(
#     embedding[:, 0],
#     embedding[:, 1],
#     c=label_np,
#     s=2)
# # Create a legend
# legend = plt.legend(
#     *scatter.legend_elements(),
#     title='Labels'
# )
# plt.title("UMAP, colored by labels")
# plt.show()
# plt.savefig('/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/analysis/umap_labels.pdf')
# plt.clf()

# var_names = ["central-depth", 
# "central-diversity",
#  "central-loc",
#  "fragment-diversity",
#  "fragment-entropy",
#  "fragment-mad",
#  "fragment-mean",
#  "fragment-median",
#  "fragment-ratio",
#  "fragment-stdev",
#  "mean-depth",
#  "minus-one-pos",
#  "np-amplitude",
#  "np-period",
#  "np-score",
#  "plus-minus-ratio"
#  ,"plus-one-pos",
#  "var-ratio"]
# # plot other variables
# print(scalar_norm.shape)

# for col in range(scalar_norm.shape[1]):
#     scatter = plt.scatter(
#     embedding[:, 0],
#     embedding[:, 1],
#     c = scalar_norm[:, col],
#     s = 2)
#     # Create a legend
#     legend = plt.legend(
#         *scatter.legend_elements(),
#         title=f'{var_names[col]}'
#     )
#     plt.title(f"UMAP, colored by variable {var_names[col]}")
#     plt.show()
#     plt.savefig(f'/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/analysis/umap_var{col}_{var_names[col]}.pdf')
#     plt.clf()