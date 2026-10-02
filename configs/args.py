import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Model scripts and evaluation',
        usage='mian.py [<args>] [-h | --help]',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--mode',
        choices=['retrain', 'inference'],
        required=True,
        help="Select execution mode: 'retrain', 'inference'."
    )

    parser.add_argument(
        '--loss',
        choices=['Listwise', 'MSE', 'BCE', 'Hierarchical', 'Pairwise'],
        required=True,
        help="Select loss function: 'Listwise', 'MSE', 'BCE', 'Hierarchical'."
    )

    parser.add_argument(
        '--yaml',
        type=str,
        default='default.yaml',
        help="Name to the YAML config file"
    )

    parser.add_argument(
        '--fold',
        type=int,
        default=-1,
        help="Specify a fold to train (starting from 1); train all if -1"
    )

    parser.add_argument(
        '--input',
        choices=['all', 'mut', 'exp'],
        required=True,
        help="Choose input type: 'all', 'mut', or 'exp'"
    )

    parser.add_argument(
        '--normalize',
        type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
        default=True,
        help="Whether to normalize scores per drug across samples (only used in inference). Default=True"
    )

    args = parser.parse_args(args)

    return args
