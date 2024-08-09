<div align="center">

# Llama3-S: When llama learns to listen
<a href='https://huggingface.co/collections/homebrew-research/llama3-s-669df2139f0576abc6eb7405'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://huggingface.co/collections/homebrew-research/llama3-s-669df2139f0576abc6eb7405'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-green'></a>

  <img src="images/llama-listen.jpg" width="180"/>
  <p><small>Image source: <a href="https://www.amazon.co.uk/When-Llama-Learns-Listen-Feelings/dp/1839237988">"When Llama Learns to Listen"</a></small></p>
</div>

> [!WARNING]  
> llama3-s is an on-going open research experiment in its early training runs. 
> - Join us in the  `#research` channel in [Homebrew's Discord](https://discord.com/invite/FTk2MvZwJH)
> - We livestream training runs in `#research-livestream`

> [!NOTE]  
> 2nd Aug 2024 Update: 
> - llama3-s can understand female, Australian accents, i.e. our synthetic voice data generator 😂
> - Can only process single-sound instruction data
> - Current Demo: [https://dollars-scholar-wins-antique.trycloudflare.com/](https://dollars-scholar-wins-antique.trycloudflare.com/)

## About
llama3-s is an open, ongoing research experiment to extend a text-based LLM to have native "listening" ability. We are mainly  

We are training an [early fusion](https://medium.com/@raj.pulapakura/multimodal-models-and-fusion-a-complete-guide-225ca91f6861#:~:text=3.3.,-Early%20Fusion&text=Early%20fusion%20refers%20to%20combining,fused%20representation%20through%20the%20model.) model using techniques inspired by [Meta's Chameleon paper](https://arxiv.org/abs/2405.09818). Our approach is focused on token transitivity which extends LLM's vocabulary to include sound tokens, has the potential to be extended to various input types in the future.

llama3-s is being done as an open science experiment with an open source codebase and dataset. We ~~build~~ train in public:
- [`#research`](https://discord.com/invite/FTk2MvZwJH) : for discussions, updates, and questions
- [`#research-livestream`](https://discord.com/invite/FTk2MvZwJH): see our training runs live

## Current Progress
- 2 Aug: Retrained phase 1 with llama3.1 and fixes to hyperparameters, achieving significant improvement (MMLU: 0.66 -> 0.61)
- 1 Aug: Identified typo in original training recipe, causing significant degradation (MMLU: 0.6 -> 0.2), proposed fixes.
- 30 July: Presented llama3-s progress at: [AI Training: From PyTorch to GPU Clusters](https://lu.ma/ws8t6wom?tk=wZvFmm)
- 19 July: [llama3-s-2024-07-19](https://huggingface.co/homebrewltd/llama3-s-2024-07-19) understands synthetic voice with limited results
- 1 July: [llama3-s-2024-07-08](https://huggingface.co/homebrewltd/llama3-s-2024-07-08) showed converging loss (1.7) with limited data

## Training Runs: 

We provide our fully finetuned models on Phase 1 and 2 data and the initialized model with expanded vocab.

| Date       | Model Checkpoint                                                              | Dataset                                                                                                 | Tokens | Step  | Batch Size | Loss    | Training Cost |
| ---------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------ | ----- | ---------- | ------- | ------------- |
| 19 July 24 | [llama3-s-2024-07-19](https://huggingface.co/homebrewltd/llama3-s-2024-07-19) | [Instruction-Speech-Full](https://huggingface.co/homebrew-research)                                     | 1.35B  | 1195k | 128        | 1.0     |     ~300$     |
| 1 July 24  | [llama3-s-2024-07-08](https://huggingface.co/homebrewltd/llama3-s-2024-07-08) | [Instruction-Speech-Phase-2](https://huggingface.co/datasets/homebrew-research/instruction-speech-v1.5) | 700M   | 1431k | 128        | 1.7-1.8 |     ~300$     |
| 23 July 24 | [llama3-s-init](https://huggingface.co/homebrewltd/llama3-s-init)             | [Instruction-Speech-Phase-1](https://huggingface.co/datasets/homebrew-research/instruction-speech-v1)   | 0M     | N/A   | N/A        | N/A     |               |

## Join Us

llama3-s is an open research project. We're looking for collaborators, and will likely move towards crowdsourcing speech datasets in the future. 

### Quickstart with Google Colab

Get started quickly using our Google Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VW_saWuNnOrl_nYCVksqqHpJmPQsyOOM?usp=sharing)

###  Synthetic Generation

For detailed information on synthetic generation, please refer to the [Synthetic Generation Guide](synthetic_data/README.md).

### Organize the input/output directory 
1. First Clone the Repo from github:
```
git clone --recurse-submodules https://github.com/homebrewltd/llama3-s.git
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

### Training with HF Trainer
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

### Training with Torchtune
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

## References
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
