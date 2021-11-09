MSG='''
\n\n
###########################################################################################################################################\n
\t\n\n
\t INSTALL DEPENDENCIES FOR FLOWER CLIENT
\t\n\n
###########################################################################################################################################\n
'''

# Instaling python and dev tools
sudo apt update
sudo apt upgrade -y
sudo apt install python3 python3-dev python3-pip -y
echo "Python 3.6 installed"

# Installing CUDA 10.0 and cuDNN 7
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda-10-0 -y
echo "CUDA 10.0 installed"

wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn7_7.6.5.32-1+cuda10.0_amd64.deb
wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn7-dev_7.6.5.32-1+cuda10.0_amd64.deb

sudo dpkg -i libcudnn7_7.6.5.32-1+cuda10.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.0_amd64.deb
cp -r /usr/src/cudnn_samples_v7/ ~/
cd cudnn_samples_v7/mnistCUDNN/
make clean && make
./mnistCUDNN

echo "cuDNN 7 installed"


# Installing Python dependencies
sudo apt install wget -y
wget https://raw.githubusercontent.com/rafaelaBrum/control-gpu/master/requirements_client_flower.txt
pip3 install -U pip setuptools
pip3 install -r requirements_client_flower.txt
sudo apt install unzip -y
echo "Flower client requirements installed"

# Installing fuse for GCP
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install gcsfuse -y
echo "GCSFuse installed"


echo "Machine ready! Don't forget to create a image of it"

