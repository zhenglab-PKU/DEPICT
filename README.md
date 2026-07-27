# DEPICT
DEPICT (Drug Efficacy Prediction via Integrated ConText), a sample-centric deep learning framework that integrates mutation and transcriptomic features to rank candidate therapies within each sample.

## Installation
### Requirements

- Python >= 3.10
- PyTorch 2.5.0 with CUDA 12.4 support

Create environment:

```bash
conda create -n depict python=3.10
conda activate depict
```

```bash
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Data Download

The raw training data and the gene expression and mutation data used for inference in the manuscript are available at:

https://doi.org/10.6084/m9.figshare.33058595

Download `data.zip` and extract it into the project directory:

```bash
unzip data.zip -d ./data
```

## Model Training

DEPICT can be retrained from scratch using:

```bash
python main.py --mode retrain --input exp --loss Hierarchical --yaml default.yaml
```

### Training Arguments

`--input`: Specifies the molecular features used for training. Use `exp` for gene expression data only, or use `all` for integrated gene expression and mutation data.

`--loss`: Specifies the loss function used for model optimization. Different loss functions can be selected according to experimental requirements.

`--yaml`: Specifies the training configuration file. The file can be modified to change training hyperparameters.

## Training Output

The trained model will be saved automatically:

```
./model_save/<training_time>/best.pt
```

## Model Inference

### Pretrained Model

The pretrained model weights used to reproduce the results reported in the manuscript are available at:

https://doi.org/10.6084/m9.figshare.33058595

Download `model_save.zip` and extract it into the project directory:

```bash
unzip model_save.zip -d ./model_save
```

### Run Inference

Inference can be performed using the provided pretrained model or a newly trained model.

```bash
python main.py --mode inference --input exp --loss Hierarchical --yaml inference.yaml
```

### Inference Arguments

The `--input` and `--loss` settings should match those used during training.

`--yaml`: Specifies the inference configuration file. The file can be modified to specify the inference dataset path, model checkpoint path, and other inference-related parameters.

### Output

The prediction results will be saved in:

```
./results/<training_time>_inference_results.csv
```

