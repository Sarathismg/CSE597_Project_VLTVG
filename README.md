Improving Visual Grounding with Visual-Linguistic Verification and Iterative Reasoning
========
As a part of CSE 597 Project, we reimplement the original paper with full single gpu support with small tweaks for single GPU in hyperparameters.


## Installation
1. Clone the repository.
    ```bash
    git clone https://github.com/yangli18/VLTVG
    ```

2. Install PyTorch 1.5+ and torchvision 0.6+.
    ```bash
    conda install -c pytorch pytorch torchvision
    ```

3. Install the other dependencies.
    ```bash
    pip install -r requirements.txt
    ```




## Preparation
Please refer to [get_started.md](https://github.com/yangli18/VLTVG/blob/master/docs/get_started.md) for the preparation of the datasets and pretrained checkpoints.




## Training

The following is an example of model training on the RefCOCOg dataset.
```
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/VLTVG_R50_gref.py
```
We train the model on 1 GPUs with a total batch size of 16 for 60 epochs. 
The model and training hyper-parameters are defined in the configuration file ``VLTVG_R50_gref.py``. 
We prepare the configuration files for different datasets in the ``configs/`` folder. 




## Evaluation
Run the following script to evaluate the trained model with a single GPU.
```
CUDA_VISIBLE_DEVICES=0 python test.py --config configs/VLTVG_R50_referit.py --checkpoint work_dirs/VLTVG_R50_referit/chckpoint0059.pth --batch_size_test 16 --test_split val
```


## Citation
If you find the code useful, please cite the original paper. 
```
@inproceedings{yang2022vgvl,
  title={Improving Visual Grounding with Visual-Linguistic Verification and Iterative Reasoning},
  author={Yang, Li and Xu, Yan and Yuan, Chunfeng and Liu, Wei and Li, Bing and Hu, Weiming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```




## Acknowledgement
Part of our code is based on [VLTVG](https://github.com/yangli18/VLTVG).