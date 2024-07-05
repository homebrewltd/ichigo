#!/bin/bash
# Update package lists
apt-get update

# Install libaio-dev
apt-get install -y libaio-dev

apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Set libaio environment variables if needed
if [ -f "/usr/include/libaio.h" ]; then
    export CFLAGS="-I/usr/include"
    export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
    echo "Set CFLAGS and LDFLAGS for libaio"
else
    echo "libaio headers not found in expected location. You may need to set CFLAGS and LDFLAGS manually."
fi

curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

pyenv install 3.11.0
pyenv global 3.11.0

python -m venv myenv
source myenv/bin/activate

# Install main dependencies
pip install --upgrade pip

pip install -U wheel

pip install packaging

pip install flash-attn --no-build-isolation

pip install transformers datasets deepspeed accelerate tensorboardX bitsandbytes trl peft

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

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