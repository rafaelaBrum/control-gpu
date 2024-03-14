echo '''
\n\n
###########################################################################################################################################\n
\t\n\n
\t INSTALL DEPENDENCIES FOR FLOWER CLIENT
\t\n\n
###########################################################################################################################################\n
'''

# Instaling python and dev tools
sudo apt update
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev wget libbz2-dev -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.7 python3.7-dev python3-pip python3.7-distutils python3-testresources -y
echo "Python 3.7 installed"

# Installing CUDA 10.0 and cuDNN 7
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
sudo apt update
sudo apt install cuda-10-0 -y
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
wget https://raw.githubusercontent.com/rafaelaBrum/control-gpu/devel_fl_cloudlab/requirements_client_flower.txt
sudo python3.7 -m pip install -U pip setuptools
sudo python3.7 -m pip install -r requirements_client_flower.txt
sudo apt install unzip -y
echo "Flower client requirements installed"

# Installing fuse for GCP
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install gcsfuse -y
echo "GCSFuse installed"

# Installing s3fs for AWS
sudo apt install s3fs -y
echo "s3fs-FUSE installed"

echo "Machine ready! Don't forget to create a image of it"

