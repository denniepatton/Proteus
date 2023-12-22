# run_parallel_training.smk
# Robert Patton, rpatton@fredhutch.org (Ha Lab)
# v1.0 12/21/2023

"""
# before running snakemake at Fred Hutch, do in tmux terminal:
ml snakemake/5.19.2-foss-2019b-Python-3.7.4
ml Python/3.7.4-foss-2019b-fh1

# command to run snakemake (remove -np at end when done validating):
snakemake -s run_parallel_training.smk --latency-wait 60 --keep-going --cluster-config config/cluster_slurm.yaml --cluster "sbatch -p {cluster.partition} --mem={cluster.mem} -t {cluster.time} -c {cluster.ncpus} -n {cluster.ntasks} -o {cluster.output} -J {cluster.JobName}" -j 40 -np
"""

# Load configuration files
configfile: "config/config.yaml"
configfile: "config/cluster_slurm.yaml"

# Define rules
rule all:
    input:
        expand("results/{use_set}/run_parallel_training_logs/fold-{fold}_output.txt", fold=range(1, config["folds"] + 1), use_set=config["use_set"]),
        "results/{use_set}/run_parallel_training_logs/final-run_output.txt"

rule run_fold:
    output:
        "results/{use_set}/run_parallel_training_logs/fold-{fold}_output.txt"
    params:
        partition = config["cluster_slurm"]["run_fold"]["partition"],
        gres = "gpu:" + str(config["cluster_slurm"]["run_fold"]["gpus"])
    shell:
        """
        # Load conda environment and module
        source /home/rpatton/miniconda3/etc/profile.d/conda.sh
        conda activate tfenv
        ml GCC/12.2.0 CUDA/12.0.0 cuDNN/8.8.0.121-CUDA-12.0.0

        # Run Proteus.py with sbatch
        sbatch --partition={params.partition} --gres={params.gres} -o {output} --wrap "Proteus.py --scalars_reference_path {config[scalars_reference_path]} --training_home {config[training_home]} --sample_data_path {config[sample_data_path]} --model_name {config[model_name]} --use_set {config[use_set]} --process_data {config[process_data]} --test_fold {wildcards.fold}"
        """

rule run_test:
    output:
        "results/{use_set}/run_parallel_training_logs/final-run_output.txt"
    params:
        partition = config["cluster_slurm"]["run_test"]["partition"],
        gres = "gpu:" + str(config["cluster_slurm"]["run_test"]["gpus"])
    shell:
        """
        # Load conda environment and module
        source /home/rpatton/miniconda3/etc/profile.d/conda.sh
        conda activate tfenv
        ml GCC/12.2.0 CUDA/12.0.0 cuDNN/8.8.0.121-CUDA-12.0.0

        # Run Proteus.py with sbatch
        sbatch --partition={params.partition} --gres={params.gres} -o {output} --wrap "Proteus.py --scalars_reference_path {config[scalars_reference_path]} --training_home {config[training_home]} --sample_data_path {config[sample_data_path]} --model_name {config[model_name]} --use_set {config[use_set]} --process_data {config[process_data]} --test_fold 0"
        """