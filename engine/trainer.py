# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging

from ignite.engine import Events,Engine
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import Accuracy, Loss, RunningAverage

from apex import amp
import torch

from utils import batchify_ray, vis_density
import numpy as np
import os


img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).cuda())

def create_supervised_trainer(model, optimizer, loss_fn, use_cuda=True, coarse_stage = 0,swriter = None):

    if use_cuda:
        model.cuda()
    
    def _update(engine, batch):

        model.train()
        optimizer.zero_grad()
        

        rays, rgbs, bboxes,  near_fars, frame_ids = batch
        
        rays = rays[0].cuda()
        rgbs = rgbs[0].cuda()
        bboxes = bboxes[0].cuda()
        near_fars = near_fars[0].cuda()
        frame_ids = frame_ids[0]
        
        
        if engine.state.epoch<coarse_stage:
            stage2, stage1,ray_mask = model( rays, bboxes,True, near_far=near_fars)
        else:
            stage2, stage1,ray_mask = model( rays, bboxes,False, near_far = near_fars)


        if ray_mask is not None:
            #print(torch.sum(ray_mask))
            loss1 = loss_fn(stage2[0][ray_mask], rgbs[ray_mask])
            loss2 = loss_fn(stage1[0][ray_mask], rgbs[ray_mask])
        else:
            loss1 = loss_fn(stage2[0], rgbs)
            loss2 = loss_fn(stage1[0], rgbs)

            psnr1 = mse2psnr(img2mse(stage1[0], rgbs))
            psnr2 = mse2psnr(img2mse(stage2[0], rgbs))
        
        loss = loss1 + loss2

        if engine.state.epoch<coarse_stage:
            loss = loss1
        else:
            loss = loss1 + loss2


        
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()

        loss.backward()

        optimizer.step()

        iters = engine.state.iteration

        if iters % 50 ==0:
            swriter.add_scalar('Loss/train_loss',loss.item(), iters)
            swriter.add_scalar('Loss/psnr2', psnr2.item(), iters)
            swriter.add_scalar('Loss/psnr1', psnr1.item(), iters)

        return loss.item()

    return Engine(_update)





def evaluator(val_dataset, model, loss_fn, swriter, epoch):
    model.eval()


    rays, rgbs, bboxes, color, mask, ROI,near_far,frame_id = val_dataset.__getitem__(0)

    rays = rays.cuda()
    rgbs = rgbs.cuda()
    bboxes = bboxes.cuda()
    color_gt = color.cuda()
    mask = mask.cuda()
    ROI = ROI.cuda()
    near_far = near_far.cuda()
    
    with torch.no_grad():

        stage2, stage1,_ = batchify_ray(model, rays, bboxes,near_far = near_far)

        color_1 = stage2[0]
        depth_1 = stage2[1]
        acc_map_1 = stage2[2]


        color_0 = stage1[0]
        depth_0 = stage1[1]
        acc_map_0 = stage1[2]




        color_img = color_1.reshape( (color_gt.size(1), color_gt.size(2), 3) ).permute(2,0,1)
        depth_img = depth_1.reshape( (color_gt.size(1), color_gt.size(2), 1) ).permute(2,0,1)
        depth_img = (depth_img-depth_img.min())/(depth_img.max()-depth_img.min())
        acc_map = acc_map_1.reshape( (color_gt.size(1), color_gt.size(2), 1) ).permute(2,0,1)


        color_img_0 = color_0.reshape( (color_gt.size(1), color_gt.size(2), 3) ).permute(2,0,1)
        depth_img_0 = depth_0.reshape( (color_gt.size(1), color_gt.size(2), 1) ).permute(2,0,1)
        depth_img_0 = (depth_img_0-depth_img_0.min())/(depth_img_0.max()-depth_img_0.min())
        acc_map_0 = acc_map_0.reshape( (color_gt.size(1), color_gt.size(2), 1) ).permute(2,0,1)



        
        swriter.add_image('GT', color_gt, epoch)

        swriter.add_image('stage2/rendered', color_img, epoch)

        swriter.add_image('stage2/depth', depth_img, epoch)
        swriter.add_image('stage2/alpha', acc_map, epoch)

        swriter.add_image('stage1/rendered', color_img_0, epoch)
        swriter.add_image('stage1/depth', depth_img_0, epoch)
        swriter.add_image('stage1/alpha', acc_map_0, epoch)


        color_img = color_img*((mask*ROI).repeat(3,1,1))
        color_gt = color_gt*((mask*ROI).repeat(3,1,1))


        return loss_fn(color_img, color_gt).item()



def global_step_from_engine(engine):
    def wrapper(_, event_name=None):
        return engine.state.iteration

    return wrapper


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        swriter,
        resume_iter = 0
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = cfg.OUTPUT_DIR
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("RFRender.%s.train" % cfg.OUTPUT_DIR.split('/')[-1])
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, coarse_stage= cfg.SOLVER.COARSE_STAGE,swriter=swriter)




    checkpointer = ModelCheckpoint(output_dir, 'rfnr', n_saved=10, require_empty=False)
    checkpointer._iteration = resume_iter

    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer,
                                                                     'scheduler':scheduler})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')

    def val_vis(engine):
        avg_loss = evaluator(val_loader, model, loss_fn, swriter,engine.state.iteration)
        logger.info("Validation Results - Epoch: {} Avg Loss: {:.3f}"
                    .format(engine.state.epoch,  avg_loss)
                    )
        swriter.add_scalar('Loss/val_loss',avg_loss, engine.state.epoch)

        #xyz, density = vis_density(model)

        #res = torch.cat([xyz[0],density[0]],dim=1).detach().cpu().numpy()
        #np.savetxt(os.path.join(output_dir,'voxels_%d.txt' % engine.state.epoch),res)

    @trainer.on(Events.STARTED)
    def resume_training(engine):
        if resume_iter>0:
            engine.state.iteration = resume_iter
            engine.state.epoch = resume_iter // len(train_loader) 

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        
        if iter % log_period == 0:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3e} Lr: {:.2e} Speed: {:.1f}[rays/s]"
                        .format(engine.state.epoch, iter, len(train_loader), engine.state.metrics['avg_loss'], lr,float(cfg.SOLVER.BUNCH) / timer.value()))
        if iter % 1000 == 1:
            val_vis(engine)

        scheduler.step()


    #@trainer.on(Events.EPOCH_COMPLETED)
    #def adjust_learning_rate(engine):
    #    scheduler.step()


    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[rays/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            float(cfg.SOLVER.BUNCH) / timer.value()))
        timer.reset()

    if val_loader is not None:
        @trainer.on(Events.EPOCH_COMPLETED )
        def log_validation_results(engine):
            val_vis(engine)
            pass

            

    trainer.run(train_loader, max_epochs=epochs)
