
import argparse
import os
import sys
from os import mkdir
from apex import amp
import shutil




import torch.nn.functional as F

sys.path.append('..')
from config import cfg
from data import make_data_loader, make_data_loader_view
from engine.trainer import do_train
from modeling import build_model
from solver import make_optimizer, WarmupMultiStepLR,build_scheduler
from layers import make_loss

from utils.logger import setup_logger

from torch.utils.tensorboard import SummaryWriter
import torch
from layers.RaySamplePoint import RaySamplePoint
import random

torch.cuda.set_device(int(sys.argv[1]))


if len(sys.argv)>2:
    training_folder = sys.argv[2]
    assert os.path.exists(training_folder), 'training_folder does not exist.'
    cfg.merge_from_file(os.path.join(training_folder,'configs.yml'))
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR
    writer = SummaryWriter(log_dir=output_dir)

else:
    cfg.merge_from_file('../configs/config.yml')
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR
    writer = SummaryWriter(log_dir=output_dir)
    shutil.copy('../configs/config.yml', os.path.join(cfg.OUTPUT_DIR,'configs.yml'))




writer.add_text('OUT_PATH', output_dir,0)
logger = setup_logger("RFRender", output_dir, 0)
logger.info("Running with config:\n{}".format(cfg))





train_loader, dataset = make_data_loader(cfg, is_train=True)
val_loader, dataset_val = make_data_loader_view(cfg, is_train=False)
model = build_model(cfg).cuda()

#maxs = torch.max(dataset.bbox[0], dim=0).values.cuda()+3
#mins = torch.min(dataset.bbox[0], dim=0).values.cuda()-3
#model.set_max_min(maxs,mins)


optimizer = make_optimizer(cfg, model)

#scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
#                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

scheduler = build_scheduler(optimizer, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.START_ITERS, cfg.SOLVER.END_ITERS, cfg.SOLVER.LR_SCALE)

if(len(cfg.MODEL.DEVICE_IDS)>1):
    torch.cuda.set_device(cfg.MODEL.DEVICE_IDS[0])
    model = torch.nn.DataParallel(model, device_ids=cfg.MODEL.DEVICE_IDS)
else:
    torch.cuda.set_device(cfg.MODEL.DEVICE_IDS[0])


iter = 0
if len(sys.argv)>2:
    iter = int(sys.argv[3])
    para_file = 'rfnr_model_%d.pth' % iter
    model.load_state_dict(torch.load(os.path.join(training_folder,para_file),map_location='cpu'))
    para_file = 'rfnr_optimizer_%d.pth' % iter
    optimizer.load_state_dict(torch.load(os.path.join(training_folder,para_file),map_location='cpu'))
    para_file = 'rfnr_scheduler_%d.pth' % iter
    scheduler.load_state_dict(torch.load(os.path.join(training_folder,para_file),map_location='cpu'))

    logger.info("load checkpoint:{}  iter:{}".format(training_folder,iter))


loss_fn = make_loss(cfg)

# model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

do_train(
        cfg,
        model,
        train_loader,
        dataset_val,
        optimizer,
        scheduler,
        loss_fn,
        writer,
        resume_iter = iter
    )
