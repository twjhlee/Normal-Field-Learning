# Normal Field Learning

## Installation - CUDA 11.1
#### Pytorch installation
<pre><code>pip3 install torch===1.8.1+cu111 torchvision===0.9.1+cu111 torchaudio -f https://download.pytorch.org/whl/torch_stable.html</code></pre>

#### Pytorch scatter installation
<pre><code>pip install torch-scatter -f https://data.pyg.org/whl/torch-1.8.1+cu111.html</code></pre>
##### See https://data.pyg.org/whl/ for supported version of pytorch-CUDA

#### DVGO installation
<pre><code>git@github.com:twjhlee/Normal-Field-Learning.git
pip install -r requirements.txt</code></pre>
    

## Example
#### Training, testing
<pre><code>python run_all.py --config "path_to_config" </code></pre>

#### Pointcloud(ply) conversion
<pre><code>python tools/vis_volume.py "path to scene_mesh.npz" "threshold[0, 1]" </code></pre>
