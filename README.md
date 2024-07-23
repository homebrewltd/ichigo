<div align="center">

# Official repo for "Llama3-S: A Speech Multimodal Model That Natively Understanding Audio and Text Input"
<a href='https://huggingface.co/collections/jan-hq/jan-llama3-668e4dad446c8736208dca4f'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://huggingface.co/collections/jan-hq/jan-llama3-668e4dad446c8736208dca4f'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-green'></a>

  <img src="images/llama3S.webp" width="180"/>
</div>

The framework supports continual training of Meta's Llama3 models with an extended vocabulary that includes unique sound tokens, enabling the model to natively understand audio. Furthermore, we provide the codebase for synthetic single-turn sound instruction data, derived from a variety of high-quality text-only sources such as Open Hermes,...
## Contents
- [Models](#models)
- [Dataset](#dataset)
- [Synthetic Generation](#https://github.com/janhq/llama3-s/blob/main/synthetic_data/synthetic.md)
- [Folder Structure Organize](#organize-the-inputoutput-directory)
- [Training with HF Trainer](#training-with-hf-trainer)
- [Training with Torchtune](#training-with-torchtune)

## Models:

We provide both our fully finetuned models on Phase 1 and 2 data. For detailed instructions on how to use these models, please refer to the guidelines provided in the Hugging Face link:
| Date       | HF Checkpoint                                   | Tokens | Step | Loss |
|------------|-------------------------------------------------|--------|------|--------------- |
| 📅 2023-12-11 | 🔗 [Llama3-S-Phase-2](https://huggingface.co/jan-hq/Jan-Llama3s-cp-6520-intermediate) | 🔢 1.35B | 🔄 1195k | 📉 1.7-1.8 |
| 📅 2024-12-28 | 🔗 [Llama3-S-Phase-1](homebrew-research/llama3-s-0708) | 🔢 700M | 🔄 1431k | 📉 1.0 |


## Dataset

We provide 3 different version of the processed data for model training, converted to the Llama3 format and ready for fine-tuning:
| Date       | HF Checkpoint                                   | Tokens | 
|------------|-------------------------------------------------|--------|
| 📅 2024-07-19 | 🔗 [Instruction-Speech-Full](https://huggingface.co/PY007/TinyLlama-1.1B-Chat-v0.1) | 🔢 1.35B | 
| 📅 2024-07-18 | 🔗 [Instruction-Speech-Phase-2](https://huggingface.co/PY007/TinyLlama-1.1B-Chat-v0.3) | 🔢 800M |
| 📅 2024-06-30 | 🔗 [Instruction-Speech-Phase-1](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.4) | 🔢 450M |

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
      journal={arXiv preprint}
}

@misc{zhang2024adamminiusefewerlearning,
      title={Adam-mini: Use Fewer Learning Rates To Gain More}, 
      author={Yushun Zhang and Congliang Chen and Ziniu Li and Tian Ding and Chenwei Wu and Yinyu Ye and Zhi-Quan Luo and Ruoyu Sun},
      year={2024},
      eprint={2406.16793},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      journal={arXiv preprint}
}

@misc{defossez2022highfi,
  title={High Fidelity Neural Audio Compression},
  author={Défossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi},
  year={2022},
  eprint={2210.13438},
  archivePrefix={arXiv},
  journal={arXiv preprint}
}
```
## Acknowledgement

- [Torchtune](https://github.com/pytorch/torchtune): The codebase we built upon
- [Llama3](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6): the Family of Models that we based on that has the amazing language capabilities !!!
