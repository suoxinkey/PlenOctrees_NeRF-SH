# encoding: utf-8

from .rfrender import RFRender


def build_model(cfg):
    model = RFRender()
    return model
