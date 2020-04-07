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


def create_supervised_trainer(model, optimizer, loss_fn, use_cuda=True, swriter = None):

    if use_cuda:
        model.cuda()

    global iters
    iters = 0


    
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        

        rays, rgbs, bboxes = batch

        rays = rays[0].cuda()
        rgbs = rgbs[0].cuda()
        bboxes = bboxes[0].cuda()
        

        color, depth = model( rays, bboxes)

        loss = loss_fn(color, rgbs)


        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        #loss.backward()

        optimizer.step()

        iters = engine.state.iteration

        if iters % 50 ==0:
            swriter.add_scalar('Loss/train_loss',loss.item(), iters)


        return loss.item()

    return Engine(_update)




def create_supervised_evaluator(model,  metrics=None, swriter = None):

    metrics = metrics or {}


    
    def _inference(engine, batch,num =0):
        model.eval()
  
        

        rays, rgbs, bboxes, color, mask, ROI = batch

        rays = rays[0].cuda()
        rgbs = rgbs[0].cuda()
        bboxes = bboxes[0].cuda()
        color_gt = color[0].cuda()
        mask = mask[0].cuda()
        ROI = ROI[0].cuda()
        
        with torch.no_grad():

            color, depth = batchify_ray(model, rays, bboxes)


            color_img = color.reshape( (color_gt.size(1), color_gt.size(2), 3) ).permute(2,0,1)
            depth_img = depth.reshape( (color_gt.size(1), color_gt.size(2), 1) ).permute(2,0,1)

            depth_img = depth_img/depth_img.max()

            num = engine.state.iteration


            swriter.add_image('vis/rendered', color_img, num)
            swriter.add_image('vis/GT', color_gt, num)
            swriter.add_image('vis/depth', depth_img, num)

            color_img = color_img*((mask*ROI).repeat(3,1,1))
            color_gt = color_gt*((mask*ROI).repeat(3,1,1))

            num = num + 1


            return (color_img, color_gt)

    engine = Engine(_inference)
    for name, metric in metrics.items():
        metric.attach(engine, name)



    return engine






def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        swriter
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = cfg.OUTPUT_DIR
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("RFRender.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, swriter=swriter)

    evaluator = create_supervised_evaluator(model, metrics={'loss': Loss(loss_fn)}, swriter=swriter)



    checkpointer = ModelCheckpoint(output_dir, 'rfnr', checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                        .format(engine.state.epoch, iter, len(train_loader), engine.state.metrics['avg_loss']))


    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_learning_rate(engine):
        scheduler.step()


    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        timer.reset()

    if val_loader is not None:
        @trainer.on(Events.EPOCH_COMPLETED )
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            avg_loss = metrics['loss']
            logger.info("Validation Results - Epoch: {} Avg Loss: {:.3f}"
                        .format(engine.state.epoch,  avg_loss)
                        )
            swriter.add_scalar('Loss/val_loss',avg_loss, engine.state.epoch)

            xyz, density = vis_density(model)

            swriter.add_mesh('density',vertices=xyz, colors=density)
            swriter.flush()

            

    trainer.run(train_loader, max_epochs=epochs)
