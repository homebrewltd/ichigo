# sound_instruct_llama3

## Organize the input/output directory 
1. First Clone the Repo from github:
```
git clone --single-branch --branch training_script https://github.com/janhq/llama3-s.git
```
The folder structure should be organized as follows before training. Note that tokenizer should be on the same directory as the model.
```
llama3-s
├── HF_Trainer
├── scripts
├── torchtune
├── model_zoo
│   ├── LLM
│   │   ├── Meta-Llama-3-8B-Instruct
│   │   ├── Jan-Llama3s-cp-6520-intermediate
│   │   ├── Meta-Llama-3-70B-Instruct

```
## Training with HF Trainer
### Install Depencencies
```
python -m venv hf_trainer
chmod +x scripts/install.sh
./scripts/install.sh
```
Restart shell now
```
chmod +x scripts/setup.sh
./scripts/setup.sh
source myenv/bin/activate
```
### Logging Huggingface
```
huggingface-cli login --token=<token>
```
### Training
```
export CUTLASS_PATH="cutlass"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch --config_file ./accelerate_config.yaml train.py 
```
## Training with Torchtune
### Install Package
```
python -m venv torchtune
pip install --pre torch==2.5.0.dev20240617  --index-url https://download.pytorch.org/whl/nightly/cu121 #or cu118
pip install --pre torchdata --index-url https://download.pytorch.org/whl/nightly
cd ./torchtune
pip install -e .
```
You can also download the model using tune:
```
tune download meta-llama/Meta-Llama-3-70b --hf-token <token> --output-dir ../model_zoo/Meta-Llama-3-70b --ignore-patterns "original/consolidated*"
```
Setup the Dataset from HF path by change the path and change the name of the model in the following YAML file.
```
nano torchtune/recipes/configs/jan-llama3-s/8B_full.yaml
```

### Training Mutil GPU (1-8GPUs Supported)
```
tune run --nproc_per_node 4 full_finetune_distributed --config janhq-llama3-s/8B_full
```