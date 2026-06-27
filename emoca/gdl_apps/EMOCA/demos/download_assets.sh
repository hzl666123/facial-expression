#!/bin/bash

cd ../../.. 
mkdir -p assets 
cd assets

echo "In order to run EMOCA, you need to download FLAME. Before you continue, you must register and agree to license terms at:"
echo -e '\e]8;;https://flame.is.tue.mpg.de\ahttps://flame.is.tue.mpg.de\e]8;;\a'

while true; do
    read -p "I have registered and agreed to the license terms at https://flame.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

echo "If you wish to use EMOCA, please register at:" 
echo -e '\e]8;;https://emoca.is.tue.mpg.de\ahttps://emoca.is.tue.mpg.de\e]8;;\a'
while true; do
    read -p "I have registered and agreed to the license terms at https://emoca.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done


echo "Downloading assets to run EMOCA..." 

# ---------- EMOCA v1 ----------
echo "Downloading EMOCA..."
mkdir -p EMOCA/models 
cd EMOCA/models 
if [ ! -f "EMOCA.zip" ]; then
    echo "EMOCA.zip not found, downloading..."
    wget https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/EMOCA.zip -O EMOCA.zip
else
    echo "EMOCA.zip already exists, skipping download."
fi
echo "Extracting EMOCA..."
unzip -o EMOCA.zip
cd ../../

# ---------- EMOCA v2 ----------
echo "Downloading EMOCA v2..."
mkdir -p EMOCA/models 
cd EMOCA/models 
for zipfile in EMOCA_v2_mp.zip EMOCA_v2_lr_mse_20.zip EMOCA_v2_lr_cos_1.5.zip; do
    if [ ! -f "$zipfile" ]; then
        echo "$zipfile not found, downloading..."
        wget "https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/$zipfile" -O "$zipfile"
    else
        echo "$zipfile already exists, skipping download."
    fi
done
echo "Extracting EMOCA v2..."
unzip -o EMOCA_v2_mp.zip
unzip -o EMOCA_v2_lr_mse_20.zip
unzip -o EMOCA_v2_lr_cos_1.5.zip
cd ../../

# ---------- DECA model (inside EMOCA/models) ----------
echo "Downloading DECA..."
mkdir -p EMOCA/models 
cd EMOCA/models 
if [ ! -f "DECA.zip" ]; then
    echo "DECA.zip not found, downloading..."
    wget https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/DECA.zip -O DECA.zip
else
    echo "DECA.zip already exists, skipping download."
fi
echo "Extracting DECA..."
unzip -o DECA.zip
cd ../../

# ---------- DECA related assets (在 assets/ 根目录) ----------
echo "Downloading DECA related assets"
if [ ! -f "DECA.zip" ]; then
    echo "DECA.zip (root) not found, downloading..."
    wget https://download.is.tue.mpg.de/emoca/assets/DECA.zip -O DECA.zip
else
    echo "DECA.zip (root) already exists, skipping download."
fi
if [ ! -f "FaceRecognition.zip" ]; then
    echo "FaceRecognition.zip not found, downloading..."
    wget https://download.is.tue.mpg.de/emoca/assets/FaceRecognition.zip -O FaceRecognition.zip
else
    echo "FaceRecognition.zip already exists, skipping download."
fi
echo "Extracting DECA related assets..."
unzip -o DECA.zip
unzip -o FaceRecognition.zip

# ---------- FLAME related assets ----------
echo "Downloading FLAME related assets"
if [ ! -f "FLAME.zip" ]; then
    echo "FLAME.zip not found, downloading..."
    wget https://download.is.tue.mpg.de/emoca/assets/FLAME.zip -O FLAME.zip
else
    echo "FLAME.zip already exists, skipping download."
fi
echo "Extracting FLAME..."
unzip -o FLAME.zip
echo "Assets for EMOCA downloaded and extracted."

# ---------- Example test data ----------
cd ../
mkdir -p data 
cd data 
echo "Downloading example test data"
if [ ! -f "EMOCA_test_example_data.zip" ]; then
    echo "EMOCA_test_example_data.zip not found, downloading..."
    wget https://download.is.tue.mpg.de/emoca/assets/data/EMOCA_test_example_data.zip -O EMOCA_test_example_data.zip
else
    echo "EMOCA_test_example_data.zip already exists, skipping download."
fi
unzip -o EMOCA_test_example_data.zip
echo "Example test data downloaded and extracted."

cd ../gdl_apps/EMOCA/demos