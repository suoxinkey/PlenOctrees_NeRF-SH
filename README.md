
# PlenOctrees_NeRF-SH
This is an implementation of the Paper [PlenOctrees for Real-time Rendering of Neural Radiance Fields](https://alexyu.net/plenoctrees/#demo-section). Not only the code provides the implementation of the NeRF-SH，but also provides the conversion code from NeRF-SH to PlenOctree. You can use the code to generate the .npz file so as to run the [C++ renderer](https://github.com/sxyu/volrend) by the [PlenOctrees for Real-time Rendering of Neural Radiance Fields](https://alexyu.net/plenoctrees/#demo-section). And the conversion code is in the tools/PlenOctrees.ipynb. The results by our code is shown in the ![Screenshot](https://github.com/suoxinkey/PlenOctrees_NeRF-SH/blob/main/img/PlenOctree.PNG).
But before using the code, you must train the NeRF-SH model. If you don't want to train the model, please concat the mail:suoxin@shanghaitech.edu.cn.


# Quick Start
The implementation of dataloader is from the [Multi-view Neural Human Rendering (NHR)](https://github.com/wuminye/NHR). So the datasets format should be the same as the[NHR](https://github.com/wuminye/NHR).        
To train the code: 
```
    
cd tools && python train_net.py <gpu id>     
```
And you can run the tools/PlenOctrees.ipynb to generate the .npz file which can run the [C++ renderer](https://github.com/sxyu/volrend) by the [PlenOctrees for Real-time Rendering of Neural Radiance Fields](https://alexyu.net/plenoctrees/#demo-section). 


# Requirements
- [yacs](https://github.com/rbgirshick/yacs) (Yet Another Configuration System)
- [PyTorch](https://pytorch.org/) (An open source deep learning platform) 
- [ignite](https://github.com/pytorch/ignite) (High-level library to help with training neural networks in PyTorch)

If you have any questions, you can contact suoxin@shanghaitech.edu.cn.
## Citation
```
@inproceedings{yu2021plenoctrees,
      title={PlenOctrees for Real-time Rendering of Neural Radiance Fields},
      author={Alex Yu and Ruilong Li and Matthew Tancik and Hao Li and Ren Ng and Angjoo Kanazawa},
      year={2021},
      booktitle={arXiv},
}

```

