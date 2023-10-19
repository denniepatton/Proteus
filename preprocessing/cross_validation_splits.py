
import os
def parse_splits(splits):
    with open(splits, 'r') as f:
        training = f.readline().strip()
        validation = f.readline().strip()
        testing = f.readline().strip()
    training = training.split(',')
    validation = validation.split(',')
    testing = testing.split(',')
    return training[1:], validation[1:], testing[1:]

def create_cross_validation_splits(split_file_path, output_file_path):
    train, val, test = parse_splits(split_file_path)
    train.extend(val)
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    for i, sample in enumerate(train):
        i_val = [sample]
        i_train = [s for s in train if s!=sample]
        with open(os.path.join(output_file_path, f"fold{i}.csv"), 'w') as f:
            f.write("Training," + ",".join(i_train))
            f.write("\n")
            f.write("Validation," + ",".join(i_val))
            f.write("\n")
            f.write("Testing," + ",".join(test))

def main():
    create_cross_validation_splits(split_file_path="/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/continuous/splits_seed206.csv",
                                   output_file_path="/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/continuous/cross_validation_seed206/")

if __name__ == "__main__":
    main()