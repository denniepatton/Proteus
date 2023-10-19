import numpy as np

PERCENT = 0.2
RANDOM_SEED = 42
training = np.load("/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/training.npz")

train_ids = training["ids"]
train_images = training["images"]
train_scalars = training["scalars"]
train_labels = training["labels"]

length = len(train_labels)
sample_size = int(PERCENT * length)
inds = np.random.choice(length, size=sample_size, replace=False)

train_ids_filtered = train_ids[inds]
train_images_filtered = train_images[inds].astype('float32')
train_scalars_filtered = train_scalars[inds].astype('float32')
train_labels_filtered = train_labels[inds]

np.savez("/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/training_small.npz",\
          ids=train_ids_filtered, images=train_images_filtered, scalars=train_scalars_filtered, labels=train_labels_filtered)