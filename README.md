# Official repo for "Llama3-S: A Speech Multimodal Model That Natively Understanding Audio and Text Input
<a href='https://huggingface.co/collections/jan-hq/jan-llama3-668e4dad446c8736208dca4f'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://huggingface.co/collections/jan-hq/jan-llama3-668e4dad446c8736208dca4f'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-green'></a>

The framework supports continual training of Meta's Llama3 models with an extended vocabulary that includes unique sound tokens, enabling the model to natively understand audio. Furthermore, we provide the codebase for synthetic single-turn sound instruction data, derived from a variety of high-quality text-only sources such as Open Hermes,...
## Contents
- [Synthetic Generation](#https://github.com/janhq/llama3-s/tree/main/synthetic_data/synthetic.md)
- [Organizing](#organize-the-inputoutput-directory)
- [Training with HF Trainer](#training-with-hf-trainer)
- [Training with Torchtune](#training-with-torchtune)

## Synthetic Generation

For detailed information on synthetic generation, please refer to the [Synthetic Generation Guide](synthetic_data/synthetic.md).
## Organize the input/output directory 
1. First Clone the Repo from github:
```
git clone --single-branch --branch training_script https://github.com/janhq/llama3-s.git
```
2. Organize the folder structure as follows before training:
```
llama3-s
├── HF_Trainer
├── synthetic_data
├── scripts
├── torchtune
├── model_zoo
│   ├── LLM
│   │   ├── Meta-Llama-3-8B-Instruct
│   │   ├── Meta-Llama-3-70B-Instruct

```
## Training with HF Trainer
1. Install Depencencies
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
2. Logging Huggingface
```
huggingface-cli login --token=<token>
```
3. Training
```
export CUTLASS_PATH="cutlass"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch --config_file ./accelerate_config.yaml train.py 
```
## Training with Torchtune
1. Install Package
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

2. Training Mutil GPU (1-8GPUs Supported)
```
tune run --nproc_per_node 4 full_finetune_distributed --config janhq-llama3-s/8B_full
```
## Reference
```bibtex
@misc{chameleonteam2024chameleonmixedmodalearlyfusionfoundation,
      title={Chameleon: Mixed-Modal Early-Fusion Foundation Models}, 
      author={Chameleon Team},
      year={2024},
      eprint={2405.09818},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.09818}, 
}

@misc{zhang2024adamminiusefewerlearning,
      title={Adam-mini: Use Fewer Learning Rates To Gain More}, 
      author={Yushun Zhang and Congliang Chen and Ziniu Li and Tian Ding and Chenwei Wu and Yinyu Ye and Zhi-Quan Luo and Ruoyu Sun},
      year={2024},
      eprint={2406.16793},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.16793}, 
}
```
## Acknowledgement

- [Torchtune](https://github.com/pytorch/torchtune): The codebase we built upon
- [Llama3](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6): the Family of Models that we based on that has the amazing language capabilities !!!
