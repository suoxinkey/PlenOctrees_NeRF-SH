import torch
import sys
sys.path.append('..')
 

from config import cfg
from data import make_data_loader
torch.cuda.set_device(3)


cfg.merge_from_file('../configs/train_mnist_softmax.yml')
cfg.freeze()


train_loader, dataset = make_data_loader(cfg, is_train=True)



for i in train_loader:
    rays, rgbs = i
    pass