from .GeoRGBD import GeoRGBD

def build_model(cfg):
    if cfg.name == "GeoRGBD":
        model = GeoRGBD(cfg)
    else:
        raise ValueError("Model {} is not recognized.".format(cfg.name))

    return model
