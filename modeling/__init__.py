# encoding: utf-8

from .rfrender import RFRender


def build_model(cfg):
    model = RFRender(cfg.MODEL.COARSE_RAY_SAMPLING, cfg.MODEL.FINE_RAY_SAMPLING, boarder_weight= cfg.MODEL.BOARDER_WEIGHT, sample_method = cfg.MODEL.SAMPLE_METHOD, same_space_net = cfg.MODEL.SAME_SPACENET,
                    TriKernel_include_input = cfg.MODEL.TKERNEL_INC_RAW, use_sh = cfg.MODEL.USE_SH)
    return model
