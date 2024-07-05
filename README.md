# sound_instruct_llama3

## Install
```
chmod +x install.sh
./install.sh
```

## Training
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch train.py --config_file ./accelerate_config.yaml
```