import shap
import os
# import tensorflow as tf
import json
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
model_path = "/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/model/_models/JULY17_RUN/REGRESSION_SCALARS_07-17-09:46"


def get_args_from_json(filepath):
    with open(filepath, 'r') as file:
        json_data = json.load(file)

    namespace = type('', (), json_data)()
    return namespace

def load_model(path, model_type):
    if model_type == "regression":
        import regression_models as m
    if model_type == "classification":
        import classification_models as m
    args = get_args_from_json(os.path.join(path, "settings.json"))
    model = m.get_model_type(args)
    model.load_weights(os.path.join(path, "weights.hdf5"))
    return model, args

def load_data(args, path='/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/continuous/training.npz', bg_size=1000):
    training_data = np.load(path)
    print("loaded training data")
    i = training_data["images"]
    s = training_data["scalars"]
    l = training_data["labels"]
    images_train, images_val, scalars_train, scalars_val, labels_train, labels_val = train_test_split(i, s, l, test_size=0.2, random_state=args.seed)
    if args.modalities == "images":
        if args.use_attention:
            model_training_inputs = [images_train[:bg_size], images_train[:bg_size]]
            model_val_inputs = [images_val[:bg_size], images_val[:bg_size]]
        else:
            model_training_inputs = images_train[:bg_size]
            model_val_inputs = images_val[:bg_size]
        del scalars_train
        del scalars_val
    elif args.modalities == "scalars":
        if args.use_attention:
            model_training_inputs = [scalars_train[:bg_size], scalars_train[:bg_size]]
            model_val_inputs = [scalars_val[:bg_size], scalars_val[:bg_size]]
        else:
            model_training_inputs = scalars_train[:bg_size]
            model_val_inputs = scalars_val[:bg_size]
        del images_train
        del images_val
    else:
        if args.use_attention:
            model_training_inputs = [images_train[:bg_size], images_train[:bg_size], scalars_train[:bg_size], scalars_train[:bg_size]]
            model_val_inputs = [images_val[:bg_size], images_val[:bg_size], scalars_val[:bg_size], scalars_val[:bg_size]]
        else:
            model_training_inputs = [images_train[:bg_size], scalars_train[:bg_size]]
            model_val_inputs = [images_val[:bg_size], scalars_val[:bg_size]]
    return model_training_inputs, model_val_inputs

def run_shap(model, background, output, feature_names=None):
    if os.path.exists(os.path.join(output, 'shap_values.pkl')):
        shap_values = pickle.load(open(os.path.join(output, 'shap_values.pkl'), 'rb'))
    else:
        print("running shap...")
        # shap.explainers.deep.deep_tf.op_handlers["AddV2"] = shap.explainers.deep.deep_tf.passthrough
        explainer = shap.GradientExplainer(model, background)
        print("created explainer...")
        # Calculate Shap values
        shap_values = explainer.shap_values(background)
        print("calculated shap values")
        pickle.dump(shap_values, open(os.path.join(output, 'shap_values.pkl'), 'wb'))
    images = np.array(shap_values[0][0])
    images_t = np.transpose(images, (2,0,1))
    means = np.mean(np.abs(images_t), axis=1)
    neg = np.where(images_t < 0, images_t, 0)
    neg = np.true_divide(neg.sum(1),(neg!=0).sum(1))
    pos = np.where(images_t > 0, images_t, 0)
    pos = np.true_divide(pos.sum(1),(pos!=0).sum(1))
    plt.clf()
    cols = ['1. Depth', '2. Probable Nucleosome Center Profile', '3. Phased Nucleosome Profile', '4. Fragment Length Short:Long ratio', '5. Fragment Length Diversity', '6. Fragement Length Entropy', '7. Peaks', '8. A-Adenine', '9. C-Cytosine', '10. G-Guanine', '11. T-Tyrosine']
    fig, axs = plt.subplots(11, 1, figsize=(8, 11), sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(pos[i], label='Positive Mean SHAP Signals')
        ax.plot(neg[i], label='Negative Mean SHAP Signals')
        ax.set_ylabel(f'{i}', rotation=0)
        
        # Set y-axis limits symmetrically around 0
        max_val = max(np.max(pos[i]), np.max(neg[i]))
        min_val = min(np.min(pos[i]), np.min(neg[i]))
        max_abs = max(abs(max_val), abs(min_val))
        ax.set_ylim(-max_abs, max_abs)
        
        # ax.legend()
    # Create a common legend outside the subplots
    # plt.legend(loc='upper right', bbox_to_anchor=(1.15, 0.5), title='Signals')
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.savefig("/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/model/_models/JULY17_RUN/REGRESSION_V1_07-17-12:37/interpretation_images.png")


    feature_arrays = []
    for i in range(images_t.shape[0]):
        plt.clf()
        a = images_t[i]
        feature_arrays.append(np.abs(a))
        g = sns.clustermap(a, cmap="seismic", center=0, row_cluster=True, col_cluster=False)
        g.ax_row_dendrogram.set_visible(False)
        g.ax_col_dendrogram.set_visible(False)
        plt.savefig(f"/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/model/_models/JULY17_RUN/REGRESSION_V1_07-17-12:37/shap_images_cluster_f{i}.png")
    plt.clf()
    fig, ax = plt.subplots(nrows=11)
    for f in feature_arrays:
        f_mean = np.mean(f, axis=0)


    # shap_values_arr = np.array(shap_values)
    # shap_values_arr.tofile(os.path.join(output, "shap.csv"), sep = ',')
    # Plot summary
    # plt.clf()
    # shap.summary_plot(shap_values, background, feature_names=feature_names, show=False)
    # plt.savefig(os.path.join(output, "interpretation.png"))

def main():
    model_path = "/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/model/_models/JULY17_RUN/REGRESSION_V1_07-17-12:37"
    model, args = load_model(model_path, "regression")
    model_training_inputs, model_val_inputs = load_data(args)
    run_shap(model, model_training_inputs, model_path)

if __name__ == "__main__":
    main()

# import regression_models
# args = get_args_from_json(os.path.join(model_path, "settings.json"))
# model = regression_models.get_model_type(args)
# model.load_weights(os.path.join(model_path, "weights.hdf5"))

# FEATURES = ["central-depth","central-diversity","central-loc",
#             "fragment-diversity","fragment-entropy","fragment-mad",
#             "fragment-mean","fragment-median","fragment-ratio",
#             "fragment-stdev","mean-depth","minus-one-pos",
#             "np-amplitude","np-period","np-score",
#             "plus-minus-ratio","plus-one-pos","var-ratio"]
# training_path = '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/continuous/training.npz'
# training_data = np.load(training_path)
# print("loaded training data")
# i = training_data["images"]
# s = training_data["scalars"]
# l = training_data["labels"]
# images_train, images_val, scalars_train, scalars_val, labels_train, labels_val = train_test_split(i, s, l, test_size=0.2, random_state=args.seed)
# # scalars_train = tf.convert_to_tensor(scalars_train)
# # scalars_val = tf.convert_to_tensor(scalars_val)
# # Create object that can calculate shap values
# # scalars_train_subset = s[:500]
# scalars_train_subset = scalars_train[:1000]
# print(scalars_train_subset)
# print("running shap...")
# # shap.explainers.deep.deep_tf.op_handlers["AddV2"] = shap.explainers.deep.deep_tf.passthrough
# explainer = shap.Explainer(model, scalars_train_subset)
# print("created explainer...")
# # Calculate Shap values
# shap_values = explainer.shap_values(scalars_train_subset)
# print("calculated shap values")
# # Plot summary
# plt.clf()
# shap.summary_plot(shap_values, scalars_train_subset, feature_names=FEATURES, show=False)
# plt.savefig(os.path.join(model_path, "interpretation.png"))

for i in range(11):
    plt.clf()
    a = images_t[i]
    g = sns.clustermap(a, cmap="seismic", center=0, row_cluster=True, col_cluster=False)
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    plt.savefig(f"/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/model/_models/JULY17_RUN/REGRESSION_V1_07-17-12:37/shap_images_cluster_f{i}.png")
