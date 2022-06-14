"""
Bonito model viewer - display a model architecture for a given config.
"""

import toml
import argparse
from bonito.util import load_symbol


def main(args):
    config = toml.load(args.config)
    Model = load_symbol(config, "Model")
    model = Model(config)
    print(model)
    print("Total parameters in model", sum(p.numel() for p in model.parameters()))

    #import torch
    #from thop import profile
    #input = torch.randn(128,1,10000)
    #flops, params = profile(model, inputs=(input, ))
    #print('FLOPs = ' + str(flops/1000**3) + 'G')
    #print('Params = ' + str(params/1000**2) + 'M')


def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("config")
    return parser
