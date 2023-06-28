# Auto-Label-AVD
[Yimeng Li](https://yimengli46.github.io/) \
George Mason University

## installation
We recommend using Python version 3.8.4 and creating a conda environment for this project. 
Please follow the steps below:
```
conda create --name auto_label_avd python=3.8.4
source activate auto_label_avd
```
Next, set up the repository environment by installing Detic, SAM, and MaskFormer. We use PyTorch 1.8.0 and Detectron2. Run the following commands:
```
# Create the top-level directory
mkdir auto_sseg_avd
cd auto_sseg_avd

# Install PyTorch and Detectron2
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html 

# Install Detic (Note: I made small changes to the Detic code, so use my repository of Detic)
git clone https://github.com/yimengli46/auto_sseg_avd_Detic.git --recurse-submodules
cd auto_sseg_avd_Detic
pip install -r requirements.txt
mkdir models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
mv Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth models/

# Install SAM
cd auto_sseg_avd
pip install opencv-python pycocotools matplotlib onnxruntime onnx
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
mkdir model_weights
mv sam_vit_h_4b8939.pth model_weights/

# Install MaskFormer
cd auto_sseg_avd
git clone https://github.com/facebookresearch/MaskFormer.git
cd MaskFormer
pip install -r requirements.txt
wget https://dl.fbaipublicfiles.com/maskformer/semantic-ade20k/maskformer_swin_large_IN21k_384_bs16_160k_res640/model_final_aefa3b.pkl
mkdir model_weights
mv model_final_aefa3b.pkl model_weights

# Install sseg-sam
cd auto_sseg_avd
git clone https://github.com/yimengli46/sseg_sam.git
```
After following all the steps, your working folder structure should look like this:
```
auto_sseg_avd/
				auto_sseg_avd_Detic/
				MaskFormer/
				segment-anything/
				sseg_sam/
```
    
