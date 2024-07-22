# sound_instruct_llama3

## Clone

```
git clone --single-branch --branch training_script https://github.com/janhq/llama3-s.git
```

## Install
```
chmod +x install.sh
./install.sh
```
Restart shell now
```
chmod +x setup.sh
./setup.sh
source myenv/bin/activate
```

## Logging Huggingface

```
huggingface-cli login --token=<token>
```

## Training
```
export CUTLASS_PATH="cutlass"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch --config_file ./accelerate_config.yaml train.py 
```
## Training with Torchtune
```
cd ./torchtune
tune run --nproc_per_node 4 full_finetune_distributed --config llama2/8B_full
```