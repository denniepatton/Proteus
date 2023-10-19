import run_model
import optuna
import classification_models
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization, Activation, \
    GlobalAveragePooling1D, Reshape, Concatenate, Attention, TimeDistributed, Lambda, MultiHeadAttention, Add
from keras.regularizers import l1, l2, l1_l2
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling1D, AveragePooling1D
import numpy as np
import tensorflow as tf
import os
import utils
import logging
def optimize_image_model(trials, output_path):
    storage = optuna.storages.RDBStorage("sqlite:///"+os.path.join(output_path, "study.db"))
    study = optuna.create_study(direction="maximize", storage=storage, load_if_exists=True, study_name="image-model-hyperparameter-tuning")
    data_path = '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/categorical/compiled_data_strict_binary.npz'
    split_path = '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/continuous/splits_seed41.csv'
    data = np.load(data_path)
    training_samples, validation_samples, testing_samples = utils.parse_splits(split_path)
    filtered_training = utils.filter_compiled_by_samples(data, training_samples)
    logging.info("Loaded training data.")
    filtered_validation = utils.filter_compiled_by_samples(data, validation_samples)
    logging.info("Loaded validation data.")
    images_train, scalars_train, labels_train = filtered_training["images"], filtered_training["scalars"], filtered_training["labels"]
    images_val, scalars_val, labels_val = filtered_validation["images"], filtered_validation["scalars"], filtered_validation["labels"]
    logging.info(f"Training on {len(labels_train)}, validating on {len(labels_val)}")
    images_train = tf.convert_to_tensor(images_train)
    images_val = tf.convert_to_tensor(images_val)
    model_training_inputs = images_train
    model_val_inputs = images_val
    labels_train_ohe = tf.one_hot(tf.convert_to_tensor(labels_train, dtype=tf.int8), depth=2)
    labels_val_ohe = tf.one_hot(tf.convert_to_tensor(labels_val, dtype=tf.int8), depth=2)
    def objective(trial: optuna.Trial):
        # hyperparameter search space
        n_filters = trial.suggest_categorical('n_filters', [8, 16, 32, 64])
        n_convolutions = trial.suggest_int('n_convolutions', 1, 4, 1)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        epochs = trial.suggest_int('epochs', 3, 50)
        conv_hidden_dim = trial.suggest_int('conv_hidden_dim', 1, 20, 1)
        batch_size = trial.suggest_int('batch_size', 32, 256, 32)
        hidden_dim = trial.suggest_int('hidden_dim', 50, 200, 50)
        pool_size = trial.suggest_int('pool_size', 2, 4, 1)

        # make output directory
        trial_output = os.path.join(output_path, f"t_{trial.number}")
        if not os.path.exists(trial_output):
            os.makedirs(trial_output)
        # define model
        logging.basicConfig(filename=os.path.join(trial_output, f"model.log"), 
                        filemode='w', 
                        level=logging.INFO,
                        force=True,
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')
        sequential_input = Input(shape=(2000, 11))
        x = Reshape((2000, 11, 1))(sequential_input)
        bp_model = Sequential([
            Flatten(),
            BatchNormalization(),
            Dense(conv_hidden_dim)
        ])
        x = TimeDistributed(bp_model)(x)
        
        x = BatchNormalization()(x)
        x = Conv1D(filters=1, kernel_size=3, activation='relu', kernel_regularizer=l1_l2(l1=0.02, l2=0.02))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=pool_size)(x)        
        for _ in range(n_convolutions-1):
            x = Conv1D(filters=n_filters, kernel_size=3, activation='relu', input_shape=(2000, 11),
                   kernel_regularizer=l1_l2(l1=0.02, l2=0.02))(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=pool_size)(x)
        x = Flatten()(x)
        x = BatchNormalization()(x)
        x = Dense(hidden_dim, activation='relu')(x)
        output = Dense(2, activation='softmax')(x)

        # Create the model
        model = Model(inputs=sequential_input, outputs=output)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='categorical_crossentropy',
                    optimizer=optimizer, 
                    metrics=['categorical_accuracy'])
        utils.plot_model(model, trial_output)
        
        logging.info(f"Training model for {epochs} epochs...")
        history = model.fit(model_training_inputs, labels_train_ohe, 
                            epochs=epochs, batch_size=batch_size, verbose=1,
                            shuffle=True, validation_data=(model_val_inputs, labels_val_ohe), 
                            callbacks=[utils.LoggerCallback(), utils.es])
        model.save_weights(os.path.join(trial_output, "weights.hdf5"))
        utils.plot_convergence(history, trial_output)
        predictions = model.predict(model_val_inputs)
        truths = labels_val_ohe.numpy()
        utils.save_predictions(predictions=predictions,
                               truths=labels_val,
                               samples=filtered_validation["samples"],
                               genes=filtered_validation["genes"],
                            output=trial_output,
                            extension="-validation")
        
        accuracy = history.history['val_categorical_accuracy'][-1]
        
        return accuracy
    
    study.optimize(objective, n_trials=trials)
    study_df = study.trials_dataframe()
    study_df.to_csv(os.path.join(output_path, "study.csv"))


def optimize_scalar_model(trials, output_path):
    data_path = '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/categorical/compiled_data_strict_binary.npz'
    split_path = '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/continuous/splits_seed41.csv'
    data = np.load(data_path)
    training_samples, validation_samples, testing_samples = utils.parse_splits(split_path)
    filtered_training = utils.filter_compiled_by_samples(data, training_samples)
    logging.info("Loaded training data.")
    filtered_validation = utils.filter_compiled_by_samples(data, validation_samples)
    logging.info("Loaded validation data.")
    images_train, scalars_train, labels_train = filtered_training["images"], filtered_training["scalars"], filtered_training["labels"]
    images_train, scalars_val, labels_val = filtered_validation["images"], filtered_validation["scalars"], filtered_validation["labels"]
    logging.info(f"Training on {len(labels_train)}, validating on {len(labels_val)}")
    images_train = tf.convert_to_tensor(images_train)
    images_val = tf.convert_to_tensor(images_val)
    model_training_inputs = images_train
    model_val_inputs = images_val
    labels_train_ohe = tf.one_hot(tf.convert_to_tensor(labels_train, dtype=tf.int8), depth=2)
    labels_val_ohe = tf.one_hot(tf.convert_to_tensor(labels_val, dtype=tf.int8), depth=2)
    def objective(trial: optuna.Trial):
        # hyperparameter search space
        tf.keras.backend.clear_session()
        learning_rate = trial.suggest_float('l1_reg_rate', 1e-5, 1e-2, log=True)
        hidden_dim_scalars = trial.suggest_int('hidden_dim_scalars', 10, 150, 10)
        epochs = trial.suggest_int('epochs', 3, 50)
        batch_size = trial.suggest_int('batch_size', 32, 256, 32)
        n_hidden = trial.suggest_int('n_hidden_layers', 0, 4, 1)

        trial_output = os.path.join(output_path, f"t_{trial.number}")
        if not os.path.exists(trial_output):
            os.makedirs(trial_output)
        # define model
        scalars_in = Input(shape=(14,))
        x = Dense(hidden_dim_scalars, activation='relu')(scalars_in)
        x = BatchNormalization()(x)
        for _ in range(n_hidden):
            x = Dense(hidden_dim_scalars, activation='relu')(x)
            x = BatchNormalization()(x)
        output = Dense(2, activation='softmax')(x)
        # Create the model
        model = Model(inputs=scalars_in, outputs=output)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='categorical_crossentropy',
                    optimizer=optimizer, 
                    metrics=['categorical_accuracy'])
        utils.plot_model(model, trial_output)
        
        logging.info(f"Training model for {epochs} epochs...")
        history = model.fit(model_training_inputs, labels_train_ohe, 
                            epochs=epochs, batch_size=batch_size, verbose=1,
                            shuffle=True, validation_data=(model_val_inputs, labels_val_ohe), 
                            callbacks=[utils.LoggerCallback(), utils.es])
        model.save_weights(os.path.join(trial_output, "weights.hdf5"))
        utils.plot_convergence(history, trial_output)
        
        
        accuracy = history.history['val_categorical_accuracy'][-1]
        
        return accuracy
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    study_df = study.trials_dataframe()
    study_df.to_csv(os.path.join(output_path, "study.csv"))

def main():
    output_path="/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/model/_models/image_model_hpt"
    logging.basicConfig(filename=os.path.join(output_path, f"hpt.log"), 
                        filemode='w+', 
                        level=logging.INFO,
                        force=True,
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')
    optimize_image_model(trials=3, output_path=output_path)
    # optimize_scalar_model(trials=20, output_path="/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/model/_models/scalar_model_hpt")

if __name__ == "__main__":
    main()
