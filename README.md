# Normal Field Learning

## Torch installation
### python 3.8 recommended, Choose one of the following cuda versions

### CUDA 11.3
#### Pytorch installation
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

#### Pytorch scatter installation
    pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html

### CUDA 11.7
#### Pytorch installation
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117


#### Pytorch scatter installation
    pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

### CUDA 10.2
#### Pytorch installation
    pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102



#### Pytorch scatter installation
    install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu102.html


##### See https://data.pyg.org/whl/ for supported version of pytorch-CUDA


## Common installation
#### DVGO installation
    git@github.com:twjhlee/Normal-Field-Learning.git
    pip install -r requirements.txt

#### OpenEXR installation
    pip install git+https://github.com/jamesbowman/openexrpython.git
    
## Surface normal estimator
All implementations are based on Bae, G., Budvytis, I., & Cipolla, R. (2021). Estimating and exploiting the aleatoric uncertainty in surface normal estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 13137-13146).

Download https://drive.google.com/file/d/1pNZ1-4iX3o4bzCkd3k5GzkVl6ae9M7eo/view?usp=drive_link to <code>surface_normal_uncertainty/experiments/nomask_noset1/models</code>


## Mask estimator
All implementations are based on Chen, L. C., Papandreou, G., Schroff, F., & Adam, H. (2017). Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587.

Download https://drive.google.com/file/d/1fH2TH6jKCkmgcNVKFj0LtpX7RbU46P_U/view?usp=drive_link to <code>deeplab/checkpoints</code>

## Example dataset
Download https://drive.google.com/file/d/1FN5Nbfv1RQ5-H-qW_bmVI-t2w-_DKHts/view?usp=drive_link to <code>data</code>
## Usage
#### Preprocessing dataset
    python preprocess.py --input_dir "path_to_input_dir"
#### Training, testing
<pre><code>python run_all.py --config "path_to_config" </code></pre>

#### Pointcloud(ply) conversion
<pre><code>python tools/vis_volume.py "path to scene_mesh.npz" "threshold[0, 1]" </code></pre>
