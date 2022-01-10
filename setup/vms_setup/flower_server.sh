echo '''
\n\n
###########################################################################################################################################\n
\t\n\n
\t INSTALL DEPENDENCIES FOR FLOWER SERVER
\t\n\n
###########################################################################################################################################\n
'''

# Instaling python and dev tools
sudo apt update
sudo apt upgrade -y
sudo apt install python3 python3-dev python3-pip -y
echo "Python 3.6 installed"

# Installing Python dependencies
sudo apt install wget -y
wget https://raw.githubusercontent.com/rafaelaBrum/control-gpu/master/requirements_server_flower.txt
pip3 install -U pip setuptools
pip3 install -r requirements_server_flower.txt
sudo apt install unzip -y
echo "Flower server requirements installed"

# Installing fuse for GCP
#export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
#echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
#curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
#sudo apt-get update
#sudo apt-get install gcsfuse -y
#echo "GCSFuse installed"

# Installing s3fs for AWS
sudo apt install s3fs -y
echo "s3fs-FUSE installed"

echo "Machine ready! Don't forget to create a image of it"

