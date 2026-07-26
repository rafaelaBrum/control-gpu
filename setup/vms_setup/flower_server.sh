# Works with Ubuntu 
echo '''
\n\n
###########################################################################################################################################\n
\t\n\n
\t INSTALL DEPENDENCIES FOR FLOWER SERVER
\t\n\n
###########################################################################################################################################\n
'''

# # Instaling python virtual environment
sudo apt update
sudo apt install python3.12-dev python3-pip python3.12-venv -y
echo "Python 3.12 virtual environment installed"

# Instaling package to execute screen
sudo apt install screen -y
echo "Screen command installed"

# Installing Python dependencies
wget https://raw.githubusercontent.com/rafaelaBrum/control-gpu/new_paper/requirements_server_flower.txt
# python3.12 -m pip install testresources cffi
python3 -m venv venv
venv/bin/python3.12 -m pip install -U pip setuptools
venv/bin/python3.12 -m pip install -r requirements_server_flower.txt
sudo apt install unzip -y
wget https://raw.githubusercontent.com/rafaelaBrum/control-gpu/new_paper/config.toml
cp config.toml .flwr/config.toml
echo "Flower server requirements installed"

# Installing fuse for GCP
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc
sudo apt update
sudo apt install gcsfuse -y
echo "GCSFuse installed"

# Installing s3fs for AWS
sudo apt install s3fs -y
echo "s3fs-FUSE installed"

echo "Machine ready! Don't forget to create a image of it"

