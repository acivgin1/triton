apt install cuda-11-1

pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

cd python
pip3 install -e .

echo """export CUDA_HOME=/usr/local/cuda-11.1
export PATH=\$PATH:\$CUDA_HOME/bin
export CPATH=\$PATH:\$CUDA_HOME/include
""" >> ~/.bashrc

source ~/.bashrc
