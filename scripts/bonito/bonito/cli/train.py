#!/usr/bin/env python3

"""
Bonito training.
"""

import os
import sys
from bonito.selectGPU import pick_gpu_lowest_memory
import torch

# I have to set num_threads, or it will use cpu 2000%
torch.set_num_threads(8)
GPUID = str(pick_gpu_lowest_memory())
print("Choose GPU", GPUID)
os.environ["CUDA_VISIBLE_DEVICES"] = GPUID

from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from importlib import import_module
import pdb

from bonito.data import load_numpy, load_script
from bonito.util import __models__, default_config, default_data
from bonito.util import load_model, load_symbol, init, half_supported
#from bonito.util import Task, SimDict
from bonito.training import load_state, Trainer

import toml
import numpy as np
from torch.utils.data import DataLoader

        
def main(args):
    print(args)

    workdir = os.path.expanduser(args.training_directory)

    if os.path.exists(workdir) and not args.force:
        print("[error] %s exists, use -f to force continue training." % workdir)
        exit(1)

    init(args.seed, args.device, (not args.nondeterministic))
    device = torch.device(args.device)

    print("[loading data]")
    try:
        train_loader_kwargs, valid_loader_kwargs = load_numpy(args.chunks, args.directory)
    except FileNotFoundError:
        train_loader_kwargs, valid_loader_kwargs = load_script(
            args.directory,
            seed=args.seed,
            chunks=args.chunks,
            valid_chunks=args.valid_chunks
        )

    loader_kwargs = {
        "batch_size": args.batch, "num_workers": 4, "pin_memory": True
    }
    # I have to set drop_last=True, or I will get RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED on some servers.
    train_loader = DataLoader(**loader_kwargs, **train_loader_kwargs, drop_last=True)
    valid_loader = DataLoader(**loader_kwargs, **valid_loader_kwargs, drop_last=True)
    
    if args.pretrained:
        dirname = args.pretrained
        if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models__, dirname)):
            dirname = os.path.join(__models__, dirname)
        config_file = os.path.join(dirname, 'config.toml')
    else:
        config_file = args.config

    # Load config
    if not args.tr:
        config = toml.load(config_file)
    else:
        raise NotImplementedError("args.tr use lite transformer as decoder. However, in this version, we do not use lite transformer.")
        from fairseq import options

        parser = options.get_training_parser()
        config, extra = options.parse_args_and_arch(parser, parse_known=True)
        sys.stderr.write('[unparsed for litetr] ' + str(extra) + '\n')

    argsdict = dict(training=vars(args))

    os.makedirs(workdir, exist_ok=True)
    if not args.tr:
        toml.dump({**config, **argsdict}, open(os.path.join(workdir, 'config.toml'), 'w'))

    print("[loading model]")
    if args.tr:
        raise NotImplementedError("args.tr use lite transformer as decoder. However, in this version, we do not use lite transformer.")
        # transformer model
        src_dict = SimDict([], 0.0, None)
        tgt_dict = SimDict(['N', 'A', 'C', 'G', 'T' ,'E'], 0, 5)
        task = Task(src_dict, tgt_dict)

        from fairseq.models.transformer_multibranch_v2 import TransformerMultibranchModel
        model = TransformerMultibranchModel.build_model(config, task)
        print(model)

        from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
        criterion = LabelSmoothedCrossEntropyCriterion(config, task)

        #if getattr(args, 'score_reference', False):
        #    from fairseq.sequence_scorer import SequenceScorer
        #    generator = SequenceScorer(tgt_dict)
        #else:
        from fairseq.sequence_generator import SequenceGenerator
        generator = SequenceGenerator(
            tgt_dict,
            beam_size=getattr(config, 'beam', 5),
            max_len_a=getattr(config, 'max_len_a', 0.15),
            max_len_b=getattr(config, 'max_len_b', 0),
            min_len=getattr(config, 'min_len', 1),
            normalize_scores=(not getattr(config, 'unnormalized', False)),
            len_penalty=getattr(config, 'lenpen', 1),
            unk_penalty=getattr(config, 'unkpen', 0),
            sampling=getattr(config, 'sampling', False),
            sampling_topk=getattr(config, 'sampling_topk', -1),
            sampling_topp=getattr(config, 'sampling_topp', -1.0),
            temperature=getattr(config, 'temperature', 1.),
            diverse_beam_groups=getattr(config, 'diverse_beam_groups', -1),
            diverse_beam_strength=getattr(config, 'diverse_beam_strength', 0.5),
            #match_source_len=getattr(config, 'match_source_len', False),
            no_repeat_ngram_size=getattr(config, 'no_repeat_ngram_size', 0),
        )

    else:
        if args.pretrained:
            print("[using pretrained model {}]".format(args.pretrained))
            model = load_model(args.pretrained, device, half=False)
        else:
            model = load_symbol(config, 'Model')(config)
        criterion = None
        generator = None

    if not args.tr and config.get("lr_scheduler"):
        sched_config = config["lr_scheduler"]
        lr_scheduler_fn = getattr(
            import_module(sched_config["package"]), sched_config["symbol"]
        )(**sched_config)
    else:
        lr_scheduler_fn = None

    trainer = Trainer(
        model, device, train_loader, valid_loader, criterion=criterion,
        use_amp=half_supported() and not args.no_amp,
        lr_scheduler_fn=lr_scheduler_fn,
        restore_optim=args.restore_optim,
        save_optim_every=args.save_optim_every,
        grad_accum_split=args.grad_accum_split,
        discrete=args.discrete,
        tr=args.tr,
        generator = generator
    )

    if (',' in args.lr):
        lr = [float(x) for x in args.lr.split(',')]
    else:
        lr = float(args.lr)
    trainer.fit(workdir, args.epochs, lr, weight_decay=float(args.wd))

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("--training_directory", default="../../models/test")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--config', default=None)
    group.add_argument('--pretrained', default="")
    parser.add_argument("--directory", type=Path, default="../../external/bonito/dna_r9.4.1/")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", default='2e-3')
    parser.add_argument("--wd", default='1e-2')
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--chunks", default=0, type=int)
    parser.add_argument("--valid-chunks", default=0, type=int)
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    parser.add_argument("--restore-optim", action="store_true", default=False)
    parser.add_argument("--nondeterministic", action="store_true", default=False)
    parser.add_argument("--save-optim-every", default=10, type=int)
    parser.add_argument("--grad-accum-split", default=1, type=int)
    parser.add_argument("--discrete", action="store_true", default=False)
    parser.add_argument("--tr", action="store_true", default=False)

    return parser
