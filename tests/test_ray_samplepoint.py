import sys
sys.path.append('..')
from layers.RaySamplePoint import RaySamplePoint
import torch
import numpy as np

rsp = RaySamplePoint()
rays = torch.from_numpy(np.array([[-0.5, 0.5, 0.5, 1, 0, 0],[-0.5, 0.5, 0.5, 0.5, 0, 0.25],[0.5, -0.5, 0.5, 0, 0.5, 0.25]]).astype(np.float)).reshape((3, 6))
bbox = np.array([[[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1], [1,0,1], [1,1,1], [0,1,1]],
                 [[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1], [1,0,1], [1,1,1], [0,1,1]],
                 [[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1], [1,0,1], [1,1,1], [0,1,1]]]).astype(np.float)
bbox = torch.from_numpy(bbox).reshape((3, 8, 3))
print(bbox)
method=None
rsp.forward(rays, bbox, method)
