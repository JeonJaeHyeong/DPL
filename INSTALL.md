## Installation

Most of the requirements of this projects are exactly the same as [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). If you have any problem of your environment, you should check their [issues page](https://github.com/facebookresearch/maskrcnn-benchmark/issues) first. Hope you will find the answer.

### Requirements:
- Python >= 3.7
- PyTorch >= 1.7 (Mine 1.8.2 (CUDA 11.1))
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV


### Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create -n dpl python=3.8
conda activate dpl

# this installs the right pip and dependencies for the fresh python
conda install -y ipython scipy h5py

# scene_graph_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python overrides gpustat gitpython ipdb graphviz tensorboardx termcolor scikit-learn numpy==1.2


# follow PyTorch installation in https://pytorch.org/get-started/locally/
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout f37fdf07367a71521bd14fec66153e0996ad128c
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/JeonJaeHyeong/DPL.git
cd DPL

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop


unset INSTALL_DIR

