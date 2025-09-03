#!/usr/bin/env bash

# ------------------------------- Disable IOMMU ------------------------------ #
sudo bash -c 'echo GRUB_CMDLINE_LINUX="amd_iommu=off" >> /etc/default/grub'
sudo grub-mkconfig -o /boot/efi/EFI/ubuntu/grub.cfg
sudo reboot

# ------------------------------ Install gh cli ------------------------------ #
type -p curl >/dev/null || (sudo apt update && sudo apt install curl -y)
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg &&
	sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg &&
	echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null &&
	sudo apt update &&
	sudo apt install gh -y

# ---------------------- Install pyenv for python sanity --------------------- #
curl https://pyenv.run | bash

# --------------------- Update the shell config for bash --------------------- #
# shellcheck disable=SC2016
{
	echo 'export PYENV_ROOT="$HOME/.pyenv"'
	echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"'
	echo 'eval "$(pyenv init -)"'
	echo 'eval "$(pyenv virtualenv-init -)"'
} >>~/.bashrc

# --------------------- Install Python build dependencies -------------------- #
sudo apt update -y &&
	sudo apt install -y make build-essential libssl-dev zlib1g-dev \
		libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
		libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# ------------------------------ Set tmux color ------------------------------ #
echo 'set -g default-terminal "screen-256color"' >>~/.tmux.conf

# ------------------------------ Install Poetry ------------------------------ #
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to the PATH
# shellcheck disable=SC2016
echo 'export PATH="$HOME/.local/bin:$PATH"' >>~/.bashrc

# Create venvs within the project
/home/ubuntu/.local/bin/poetry config virtualenvs.in-project true

# Handle temporary poetry issue
# https://github.com/python-poetry/poetry/issues/1917#issuecomment-1251667047
echo 'export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring' >>~/.bashrc

# ------------------------------- Install CUDA ------------------------------- #
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin &&
	sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 &&
	sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub &&
	sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" &&
	sudo apt update -y &&
	sudo apt install -y cuda-11-8

# shellcheck disable=SC2016
{
	echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}'
	echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}'
} >>~/.bashrc

# ------------------- Install fabricmanager (for A100 only) ------------------ #
NVIDIA_DRIVER_VERSION=$(dpkg -l | grep nvidia-driver- | cut -d ' ' -f 3 | cut -d '-' -f 3) &&
	sudo apt install -y "cuda-drivers-fabricmanager-$NVIDIA_DRIVER_VERSION" &&
	sudo service nvidia-fabricmanager start

# --------------------------- Setup instance drives -------------------------- #
mkdir ~/data
sudo mkfs -t xfs /dev/nvme1n1
sudo mount /dev/nvme1n1 ~/data
sudo chown ubuntu:ubuntu ~/data

# Put HF cache in data dir
echo 'export HF_HOME=/home/ubuntu/data/huggingface' >>~/.bashrc

# ------------------------------- Restart shell ------------------------------ #
# shellcheck disable=SC2093
exec "$SHELL"

# --------------------------------- Do things -------------------------------- #
# Login to GH
echo "$GITHUB_PAT" | gh auth login --with-token
gh auth setup-git

# Clone the repo
gh repo clone amitkparekh/CoGeLoT

# cd into the repo
cd CoGeLoT || exit 1

# Install python and deps
pyenv install
poetry env use "$(pyenv which python)"
poetry install --without lint,test
poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Create the output dir on the storage drive
mkdir /home/ubuntu/data/outputs
mkdir /home/ubuntu/data/data

# symlink the outputs dir to the repo
ln -s /home/ubuntu/data/outputs ./storage/
ln -s /home/ubuntu/data/data ./storage/
