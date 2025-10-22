

## ⚙️ Dependencies

- Python 3.10
- PyTorch 2.1.1
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

```bash
# Clone the github repo and go to the default directory 'mamba'.

conda create -n your_env_name python=3.10.13
conda activate your_env_name
# install cuda 11.8
#https://blog.csdn.net/weixin_44502754/article/details/143229508
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

#conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
#conda install packaging
pip install causal-conv1d
pip install mamba-ssm
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
pip install mamba_ssm-1.1.4+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl




## 🔗 Contents


---



## <a name="models"></a>📦 Models

| Method | Params (M) | FLOPs (G) | PSNR (dB) | SSIM | Model Zoo | Visual Results |
| :----- | :--------: | :-------: | :-------: | :--: | :-------: | :------------: |
|        |            |           |           |      |           |                |
|        |            |           |           |      |           |                |

The performance is reported on RICE2. Input and Output size of FLOPs is 3×512×512.



## <a name="training"></a>🔧 Training

- 

- Run the following scripts. The training configuration is in `options/train/`.

  ```shell
  #  Remove cloud, input=256x256, 1 GPU
  python basicsr/train.py -opt options/rice2/rice2-mamba_pos.yml
  
  # Remove cloud, input=256x256, multiple GPUs
  CUDA_VISIBLE_DEVICES=6,7  torchrun  --nproc_per_node=2  --master_port=1238 basicsr/train.py -opt options/rice2/rice2-mamba_pos.yml --launcher pytorch
  
  #old torch version use
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/rice2/rice2-mamba_pos.yml --launcher pytorch
  
  ```

- The training experiment is in `experiments/`.



## <a name="testing"></a>🔨 Testing

### 🌗 Test images with HR

- Set config in ./TEST/test.py

  Set --input_dir to test cloud folder, such as

  ```
  parser.add_argument('--input_dir', default='./datasets/cloud/clound-wy/RICE2/test/cloud', type=str, help='Directory of input images or path of single image')
  ```

  Set --input_truth_dir to test truth folder, such as

  ```
  parser.add_argument('--input_truth_dir', default='./datasets/cloud/clound-wy/RICE2/test/reference/', type=str, help='Directory of input images or path of single image')
  ```

  Set --result_dir to test truth folder, such as

  ```
  parser.add_argument('--result_dir', default='./output/rice2-result', type=str, help='Directory for restored results')
  ```

  Set weight path, such as

  ```
   weights = os.path.join('experiments/rice2_mamba_add_pos_cnn/models/', 'net_g_147000.pth')
  ```

  Set parameters path, such as

  ```
  parameters = {'inp_channels':3, 'out_channels':3, 'dim':48, 'num_blocks':[4,6,6,8], 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66, 'bias':False, 'LayerNorm_type':'WithBias'}
  ```

  

- Run the following scripts. 

  ```shell
# test
  # export CUDA_LAUNCH_BLOCKING=1
  python Test/test.py 
  ```
  
- The output is in `output/`.



## <a name="results"></a>🔎 Results

We achieved state-of-the-art performance. Detailed results can be found in the paper.





## <a name="citation"></a>📎 Citation

If you find the code helpful in your resarch or work, please cite the following paper(s).

```
fff
```



## <a name="acknowledgements"></a>💡 Acknowledgements

This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR).

