pyenv install 3.11.0
pyenv global 3.11.0

python -m venv myenv
source myenv/bin/activate

# Install main dependencies
pip install --upgrade pip

pip install -U wheel

pip install packaging

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install flash-attn --no-build-isolation

pip install transformers datasets deepspeed accelerate tensorboardX bitsandbytes trl peft

# Install nvidia-cutlass
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
pip install -e .
cd ..

# Find CUTLASS path
export CUTLASS_PATH="cutlass"

# Print environment variables for verification
echo "CFLAGS: $CFLAGS"
echo "LDFLAGS: $LDFLAGS"
echo "CUTLASS_PATH: $CUTLASS_PATH"

# Add environment variables to .bashrc for persistence
echo "Adding environment variables to .bashrc for persistence"
echo "export CFLAGS=\"$CFLAGS\"" >> ~/.bashrc
echo "export LDFLAGS=\"$LDFLAGS\"" >> ~/.bashrc
echo "export CUTLASS_PATH=\"$CUTLASS_PATH\"" >> ~/.bashrc
echo "Remember to run 'source ~/.bashrc' after the script finishes to apply changes in the current session."
