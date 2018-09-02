# OpenCV Nvidia CUDA GPU driver installation

### Environment
Ubuntu 18.04

Or

Ubuntu 18.04 on Windows 10 64 bit
https://discuss.openai.com/t/installing-openai-gym-universe-on-windows/2092


### Requirement

Python 3.6

OpenCV 3.4.0

Tensorflow 1.9

### Instalation
```bash
sudo su -
apt-get update
apt-get install -y python3-pip
pip3 install --upgrade pip
sudo apt-get install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install python3.6-dev
python3.6-config --includes
#The output would be
-I/usr/include/python3.6m -I/usr/include/python3.5m
cd /usr/include/python3.6m
mv pyconfig.h pyconfig.h.bak
cp /usr/include/x86_64-linux-gnu/python3.5m/pyconfig.h /usr/include/python3.5m/
cd /root
mkdir OpenCV-tmp
cd OpenCV-tmp
git clone https://github.com/Itseez/opencv.git
cd opencv
git checkout 3.4.0
cd ../OpenCV-tmp
git clone https://github.com/Itseez/opencv_contrib.git
cd opencv_contrib
git checkout 3.4.0

cd ../opencv
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=ON \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D OPENCV_EXTRA_MODULES_PATH=/root/OpenCV-tmp/opencv_contrib/modules \
	-D BUILD_EXAMPLES=ON ..

make -j $(nproc --all)
sudo make install
sudo ldconfig

```

Checking
```bash
python3
import cv2
cv2.__version__

```

Install tqdm
```bash
pip3 install tqdm
```

Install numpy matplotlib
```bash
pip3 install numpy
pip3 install matplotlib
pip3 install pandas
```


## Install tensorflow CPU version
```bash
pip3 install tensorflow
```

## Install tensorflow-gpu (GPU version) on Ubuntu 18.04
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-390
```

Check driver installation is success
```bash
nvidia-smi
```

Install CUDA Toolkit 9.0
https://developer.nvidia.com/cuda-90-download-archive
```bash
sudo apt install nvidia-cuda-toolkit gcc-6 g++-6
nvcc --version
```
Could use update alternatives
https://askubuntu.com/questions/26498/how-to-choose-the-default-gcc-and-g-version
```bash
sudo chmod +x cuda_9.0.176_384.81_linux.run
./cuda_9.0.176_384.81_linux.run --override
```

Install CUDNN (ensure you are registered for the NVIDIA Developer Program)
https://developer.nvidia.com/cudnn
```bash
tar -zxvf cudnn-9.0-linux-x64-v7.1.tgz

```

Move the unpacked contents to your CUDA directory
```bash
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64/
sudo cp  cuda/include/cudnn.h /usr/local/cuda-9.0/include/
```

Give read access to all users
```bash
sudo chmod a+r /usr/local/cuda-9.0/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
sudo apt-get install libcupti-dev
```

Add environment variables
```bash
export PATH=/usr/local/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
Restart your terminal before proceeding to the next step. Install tensorflow-gpu
```bash
python3 -m pip install --upgrade tensorflow-gpu
python3
import tensorflow as tf
print(tf.__version__)
```

## Reference
https://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/
http://cyaninfinite.com/tutorials/installing-opencv-in-ubuntu-for-python-3/
https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-18-04-bionic-beaver-linux
