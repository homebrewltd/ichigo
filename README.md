# sound_instruct_llama3

## Logging Huggingface

```
huggingface-cli login --token=<token>
```

## Clone

```
git clone --single-branch --branch training_script https://github.com/janhq/sound_instruct_llama3
```

## Install
```
chmod +x install.sh
./install.sh

chmod +x setup.sh
./setup.sh
```

## Training
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --config_file ./accelerate_config.yaml train.py 
```
