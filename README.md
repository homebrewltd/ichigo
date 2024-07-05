# sound_instruct_llama3

## Clone

```
git clone --single-branch --branch training_script https://github.com/janhq/sound_instruct_llama3
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
```

## Logging Huggingface

```
huggingface-cli login --token=<token>
```

## Training
```
export CUTLASS_PATH="cutlass"
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --config_file ./accelerate_config.yaml train.py 
```
