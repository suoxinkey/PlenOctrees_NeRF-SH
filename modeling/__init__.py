# encoding: utf-8

from .rfrender import RFRender


def build_model(cfg):
    model = RFRender(cfg.MODEL.COARSE_RAY_SAMPLING, cfg.MODEL.FINE_RAY_SAMPLING, sample_method = cfg.MODEL.SAMPLE_METHOD)
    return model
