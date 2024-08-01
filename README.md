<div align="center">

# Llama3-S: When llama learns to listen
<a href='https://huggingface.co/collections/homebrew-research/llama3-s-669df2139f0576abc6eb7405'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://huggingface.co/collections/homebrew-research/llama3-s-669df2139f0576abc6eb7405'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-green'></a>

  <img src="images/llama-listen.jpg" width="180"/>
  <p><small>Image source: <a href="https://www.amazon.co.uk/When-Llama-Learns-Listen-Feelings/dp/1839237988">"When Llama Learns to Listen"</a></small></p>
</div>

## Introduction
Llama3-s is an open, ongoing research project by [Homebrew](https://homebrew.ltd/) to extend an a Large Language Model (LLM) that native understands audio input. Inspired by [Meta's Chameleon paper](https://arxiv.org/abs/2405.09818), it employs an early fusion model, enabling native audio comprehension. Our approach, focused on token transitivity which extends LLM's vocabulary to include sound tokens, has the potential to be extended to various input types in the future.

The project provides a full codebase and replication instructions for synthetic data creation and training.

⚠️ Work in Progress
Llama3-s is currently under active development. Please note the following limitations:

- The model currently responds only to female voices
- It processes single-turn sound instruction data

We are continuously working to expand these capabilities.

## News
- [2024/07/19] We released [llama3-s-2024-07-19](https://huggingface.co/homebrewltd/llama3-s-2024-07-19), trained on 1.35B tokens. This model achieves a loss of 1.0.
- [2024/07/01] We released [llama3-s-2024-07-08](https://huggingface.co/homebrewltd/llama3-s-2024-07-08), trained on 700M tokens. This model achieves a loss of 1.7.
- [2024/06/23] We released [llama3-s-init](https://huggingface.co/homebrewltd/llama3-s-init), our initialized model with expanded vocabulary.

## Contents
- [Models](#models)
- [Dataset](#dataset)
- [Synthetic Generation](https://github.com/homebrewltd/llama3-s/blob/main/synthetic_data/README.md)
- [Folder Structure Organize](#organize-the-inputoutput-directory)
- [Training with HF Trainer](#training-with-hf-trainer)
- [Training with Torchtune](#training-with-torchtune)

## Quickstart with Google Colab

Get started quickly using our Google Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VW_saWuNnOrl_nYCVksqqHpJmPQsyOOM?usp=sharing)


## Models:

We provide our fully finetuned models on Phase 1 and 2 data and the initialized model with expanded vocab.
| Date | Checkpoint | Tokens | Step | Batch Size | Loss | Status |
|------|------------|--------|------|------------|------|--------|
| 📅 2024-07-19 | 🔗 [llama3-s-2024-07-19](https://huggingface.co/homebrewltd/llama3-s-2024-07-19) | 🔢 1.35B | 🔄 6520 | 💼 128 | 📉 1.0| 🚧 In progress |
| 📅 2024-07-01 | 🔗 [llama3-s-2024-07-08](https://huggingface.co/homebrewltd/llama3-s-2024-07-08) | 🔢 700M | 🔄 4320 | 💼 128 | 📉 1.7-1.8  | 🚧 In progress |
| 📅 2024-06-23 | 🔗 [llama3-s-init](https://huggingface.co/homebrewltd/llama3-s-init) | 🔢 0M | 🔄 N/A | 💼 N/A | 📉 N/A | N/A |

## Dataset

We provide 3 different version of the processed data for model training, converted to the Llama3 format and ready for fine-tuning:
| Date       | HF Checkpoint                                   | Tokens | 
|------------|-------------------------------------------------|--------|
| 📅 2024-07-19 | 🔗 [Instruction-Speech-Full](https://huggingface.co/homebrew-research) | 🔢 1.35B | 
| 📅 2024-07-18 | 🔗 [Instruction-Speech-Phase-2](https://huggingface.co/datasets/homebrew-research/instruction-speech-v1.5) | 🔢 800M |
| 📅 2024-06-30 | 🔗 [Instruction-Speech-Phase-1](https://huggingface.co/datasets/homebrew-research/instruction-speech-v1) | 🔢 450M |

## Synthetic Generation

For detailed information on synthetic generation, please refer to the [Synthetic Generation Guide](synthetic_data/README.md).

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

@misc{WhisperSpeech,
      title={WhisperSpeech: An Open Source Text-to-Speech System Built by Inverting Whisper}, 
      author={Collabora and LAION},
      year={2024},
      url={https://github.com/collabora/WhisperSpeech},
      note={GitHub repository}
}
```
## Acknowledgement

- [Torchtune](https://github.com/pytorch/torchtune): The codebase we built upon
- [Accelerate](https://github.com/huggingface/accelerate): Library for easy use of distributed training
- [WhisperSpeech](https://github.com/collabora/WhisperSpeech): Text-to-speech model for synthetic audio generation 
- [Encodec](https://github.com/facebookresearch/encodec): High-fidelity neural audio codec for efficient audio compression
- [Llama3](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6): the Family of Models that we based on that has the amazing language capabilities !!!
