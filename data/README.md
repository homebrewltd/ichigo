# Pipeline to generate sound instruction dataset

## 1. Aggregate and combine available opensource dataset

- List of used opensource dataset:
  - Intel/orca_dpo_pairs
  - routellm/gpt4_dataset
  - nomic-ai/gpt4all-j-prompt-generations
  - microsoft/orca-math-word-problems-200k
  - allenai/WildChat-1M
  - Open-Orca/oo-gpt4-200k
  - Magpie-Align/Magpie-Pro-300K-Filtered
  - qiaojin/PubMedQA
  - Undi95/Capybara-ShareGPT
  - HannahRoseKirk/prism-alignment - utterances
  - BAAI/Infinity-Instruct
- Those are combined and avaialble at [jan-hq/instruction-speech-v1.5](https://huggingface.co/datasets/jan-hq/instruction-speech-v1.5)

## 2. Usage

- Convert `instruction-speech-v1.5` into sound instruct dataset

```bash
python run_tts_convert.py --total_samples <total_samples> --batch_size <batch_size>
```
