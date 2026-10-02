import os

from configs.args import parse_args
from configs.configs import Config
from scripts.inference import inference
from scripts.train import train_fold
from utils.utils import set_seed, init_logger


def main(args):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    config = Config(root_dir, args.yaml)
    config.update('root_dir', root_dir)

    set_seed(config.seed)

    logger, timestamp = init_logger(config.root_dir)

    if args.mode == 'retrain':
        model_save_dir = os.path.join(root_dir, 'model_save', timestamp)
        os.makedirs(model_save_dir, exist_ok=True)
        config.update('model_save_dir', model_save_dir)

        logger.info("======= ARGS =======")
        for k, v in vars(args).items():
            logger.info(f"{k}: {v}")

        logger.info("======= CONFIG =======")
        for k, v in vars(config).items():
            logger.info(f"{k}: {v}")

        start_fold = 1
        end_fold = config.kfold

        if args.fold != -1:
            start_fold = args.fold
            end_fold = args.fold

        for fold in range(start_fold, end_fold + 1):

            if args.fold == 6:
                logger.info(f'Fold {fold} uses all samples for training without cross-validation.')

            fold_dir = os.path.join(str(model_save_dir), f'fold_{fold}')
            os.makedirs(fold_dir, exist_ok=True)
            config.update('fold_dir', fold_dir)

            logger.info(f'Starting Fold {fold}/{end_fold}')

            train_fold(args, config, fold)

    elif args.mode == 'inference':
        inference(args, config, timestamp)

    else:

        raise ValueError("Please specify one of --retrain, --inference.")


if __name__ == "__main__":
    main(parse_args())
