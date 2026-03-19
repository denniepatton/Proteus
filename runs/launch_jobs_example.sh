#!/bin/bash

# Create a logs directory if it doesn't exist
# mkdir -p logs

# Loop over test_fold values (1 to 5) and submit separate jobs {1..5}
for test_fold in {1..5}; do
    job_script="job_fold_${test_fold}.sh"
    cat > ${job_script} <<EOL
#!/bin/bash
#SBATCH --job-name=PT_F${test_fold}  # Unique job name for each fold
#SBATCH --output=logs/PT_F${test_fold}_%j.out  # Output log file
#SBATCH --error=logs/PT_F${test_fold}_%j.err   # Error log file
#SBATCH --gpus=rtx2080ti:1
#SBATCH --time=120:00:00
#SBATCH -A nelson_p

eval "\$(micromamba shell hook -s bash)"
micromamba activate proteus
python Proteus.py --run_mode train --experiment config/experiments/biodvn_train_ProteusV1HPs.yaml --test_fold ${test_fold}

EOL

    sbatch ${job_script}
    rm -f ${job_script}
done
