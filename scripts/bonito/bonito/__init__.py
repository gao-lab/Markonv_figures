from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from bonito.cli import basecaller, train, evaluate, view, convert, download, export, selModel
import sys

modules = [
    'basecaller', 'train', 'evaluate', 'view', 'convert', 'download', 'export', 'selModel'
]

__version__ = '0.5.1'


def main():
    parser = ArgumentParser(
        'bonito',
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-v', '--version', action='version',
        version='%(prog)s {}'.format(__version__)
    )

    subparsers = parser.add_subparsers(
        title='subcommands', description='valid commands',
        help='additional help', dest='command'
    )
    subparsers.required = True

    for module in modules:
        mod = globals()[module]
        p = subparsers.add_parser(module, parents=[mod.argparser()])
        p.set_defaults(func=mod.main)

    args, unparsed = parser.parse_known_args()
    args.func(args)
