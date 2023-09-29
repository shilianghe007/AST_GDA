import argparse



def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def classifier_defaults():
    """
    Defaults for classifier.
    """
    res = dict(
        bottleneck_dim=256,
        scratch=False,
        lr=0.001,
        lr_gamma=0.001,
        lr_decay=0.75,
        momentum=0.9,
        weight_decay=1e-3,
        no_pool=False,
    )
    return res

def dataset_defaults():
    res = dict(
        root='data/mnist',
        log='logs',
        data='mnist',
        num_classes=10,
        train_resizing='default',
        val_resizing='default',
        no_hflip=False, # NOTE: trun off after testing!
        norm_mean=(0.485, 0.456, 0.406),
        norm_std=(0.229, 0.224, 0.225),
        scale=[0.08, 1.0],
        ratio=[3. / 4., 4. / 3.],
        workers=2,
    )
    return res

def conformal_defaults():
    res = dict(
        alpha = 0.05,
    )
    return res


def create_args(special_args):
    '''
    parse the special_args into the default arguments
    return the args:dict
    '''
    default = dict()
    default.update(classifier_defaults())  # classifier arguments
    default.update(dataset_defaults())  # dataset arguments
    default.update(conformal_defaults())  # conformal arguments
    default.update(special_args)  # special arguments
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default)
    return parser