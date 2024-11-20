# modular-Lan-MIA
modular-Lan-MIA is a modular python library for benchmarking membership inference attacks with several NLP models and datasets.

# Example (run_main.sh bash script)

```
#!/bin/bash

#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=4          # CPU cores per task for data loading, etc.
#SBATCH --mem=64G                  # Memory per node
#SBATCH --time=14:00:00            # Time limit
#SBATCH --partition=gpu            # GPU partition (adjust to your cluster's GPU partition)
#SBATCH --output=<output_path>.log

# Load necessary modules
module load python3                # Load Python
module load cuda/11.1              # Load CUDA toolkit

source <environment_location>/bin/activate

export OMP_NUM_THREADS=16

python3 main.py --dataset IMDb --num-epochs 10 --attack-name prediction_loss_based_mia 

```

# Input options

| Option | Description | Default |
| --- | --- | --- |
| `--dataset` | specifies a dataset on which an attack is to be performed | IMDb |
| `--model-name`| name of the pretrained NLP model | roberta-base |
| `--attack-name` | name of an attack to be performed | prediction_loss_based_mia |
| `--num-trials` | number of trials of an attack to be performed | 500 |
| `--load-model-dir` | directory from which a trained model is loaded | None |
| `--num-epochs` | number of epochs used in the fine-tuning of the model | 5 |
| `--max-sequence-length` | maximum sequence length | 512 |
| `--torch-num-threads` | number of CPU threads PyTorch uses for parallel operations | 1 |

# Evaluation metric 

Once {num-trials} attacks are performed we summarize theit effectivness through the following evaluation metrics:
- Accuracy
- Precision
- Recall
- F1
- FPR
- Advantage
            