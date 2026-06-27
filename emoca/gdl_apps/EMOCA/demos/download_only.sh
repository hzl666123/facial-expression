#!/bin/bash

# 进入项目根目录下的 assets 文件夹
cd ../../.. 
mkdir -p assets 
cd assets

# ---------- 许可证确认部分保持不变 ----------
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

echo "Starting downloads (no extraction will be performed)..."

# 1. EMOCA v1 模型
mkdir -p EMOCA/models
cd EMOCA/models
if [ ! -f "EMOCA.zip" ]; then
    wget https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/EMOCA.zip -O EMOCA.zip
else
    echo "EMOCA.zip already exists, skipping."
fi
cd ../../

# 2. EMOCA v2 模型（三个文件）
mkdir -p EMOCA/models
cd EMOCA/models
for zipfile in EMOCA_v2_mp.zip EMOCA_v2_lr_mse_20.zip EMOCA_v2_lr_cos_1.5.zip; do
    if [ ! -f "$zipfile" ]; then
        wget "https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/$zipfile" -O "$zipfile"
    else
        echo "$zipfile already exists, skipping."
    fi
done
cd ../../

# 3. DECA 模型（放在 EMOCA/models 内）
mkdir -p EMOCA/models
cd EMOCA/models
if [ ! -f "DECA.zip" ]; then
    wget https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/DECA.zip -O DECA.zip
else
    echo "DECA.zip (inside EMOCA/models) already exists, skipping."
fi
cd ../../

# 4. DECA 相关资源（两个文件，放在 assets/ 根目录）
if [ ! -f "DECA.zip" ]; then
    wget https://download.is.tue.mpg.de/emoca/assets/DECA.zip -O DECA.zip
else
    echo "DECA.zip (root) already exists, skipping."
fi
if [ ! -f "FaceRecognition.zip" ]; then
    wget https://download.is.tue.mpg.de/emoca/assets/FaceRecognition.zip -O FaceRecognition.zip
else
    echo "FaceRecognition.zip already exists, skipping."
fi

# 5. FLAME 相关资源
if [ ! -f "FLAME.zip" ]; then
    wget https://download.is.tue.mpg.de/emoca/assets/FLAME.zip -O FLAME.zip
else
    echo "FLAME.zip already exists, skipping."
fi

# 6. 示例测试数据
mkdir -p data
cd data
if [ ! -f "EMOCA_test_example_data.zip" ]; then
    wget https://download.is.tue.mpg.de/emoca/assets/data/EMOCA_test_example_data.zip -O EMOCA_test_example_data.zip
else
    echo "EMOCA_test_example_data.zip already exists, skipping."
fi
cd ../

echo "All downloads completed. No files were extracted."
echo "You can now manually unzip the files as needed."