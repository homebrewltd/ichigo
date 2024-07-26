python -m venv torchtune
pip install -U "huggingface_hub[cli]"
pip install --pre torch==2.5.0.dev20240617  --index-url https://download.pytorch.org/whl/nightly/cu121 #or cu118
pip install --pre torchdata --index-url https://download.pytorch.org/whl/nightly
cd ./torchtune
pip install -e .
tune download homebrewltd/Llama3.1_s_8b_init --hf-token <token> --output-dir ../model_zoo/Llama3.1_s_8b_init --ignore-patterns "original/consolidated*"
# huggingface_hub download hf-hub/dataset --local-dir ../model_zoo/