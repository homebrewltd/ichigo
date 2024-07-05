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