"""entry point for training a classifier"""
import argparse
import importlib
import json
import logging
import os
import pprint
import sys

import dill
import torch
import wandb
from box import Box
from torch.utils.data import DataLoader

from lib.utils import logging as logging_utils, optimizer as optimizer_utils, os as os_utils
from src.common.dataset import get_dataset
from src.trainer.binary_clf_trainer import BinaryClfTrainer
from src.trainer.regression_trainer import RegressionTrainer


def parser_setup():
    # define argparsers
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action='store_true', help="debug mode on or off")
    parser.add_argument("--config", "-c", required=False, help="config file")
    parser.add_argument("--seed", required=False, type=int, help="set the seed")

    str2bool = os_utils.str2bool
    listorstr = os_utils.listorstr
    parser.add_argument("--wandb.use", required=False, type=str2bool, default=False)
    parser.add_argument("--wandb.run_id", required=False, type=str)
    parser.add_argument("--wandb.watch", required=False, type=str2bool, default=False)
    parser.add_argument("--project", required=False, type=str, default="wandb project name")
    parser.add_argument("--exp_name", required=True)

    parser.add_argument(
        "--device", required=False,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--result_folder", "-r", required=False)
    parser.add_argument(
        "--mode", required=False, nargs="+", choices=["test", "train"],
        default=["test", "train"]
    )
    parser.add_argument("--statefile", "-s", required=False, default=None)

    # data arguments
    parser.add_argument("--data.name", "-d", required=False, choices=["ukbb_brain_age", "adni_ad"])
    parser.add_argument("--data.root_path", default=None, type=str)
    parser.add_argument("--data.train_csv", default=None, type=str)
    parser.add_argument("--data.valid_csv", default=None, type=str)
    parser.add_argument("--data.test_csv", default=None, type=str)
    parser.add_argument(
        "--data.train_num_sample", default=-1, type=int,
        help="control number of training samples"
    )
    parser.add_argument(
        "--data.frame_dim", default=1, type=int, choices=[1, 2, 3],
        help="choose which dimension we want to slice, 1 for sagittal, "
             "2 for coronal, 3 for axial"
    )
    parser.add_argument(
        "--data.frame_keep_style", default="random", type=str,
        choices=["random", "ordered"],
        help="style of keeping frames when frame_keep_fraction < 1"
    )
    parser.add_argument(
        "--data.frame_keep_fraction", default=0, type=float,
        help="fraction of frame to keep (usually used during testing with missing "
             "frames)"
    )
    parser.add_argument(
        "--data.impute", default="drop", type=str,
        choices=["drop", "fill", "zeros", "noise"]
    )

    # model related arguments
    parser.add_argument("--model.name", required=False, choices=["regression", "binary_classifier"])

    parser.add_argument("--model.arch.file", required=False, type=str, default=None)
    # 3D CNN related args
    # none

    # 2D lstm related argument
    # also uses frame dim from data args
    parser.add_argument("--model.arch.lstm_feat_dim", required=False, type=int, default=2)
    parser.add_argument("--model.arch.lstm_latent_dim", required=False, type=int, default=128)

    # 2D Slice CNN related arguments
    parser.add_argument("--model.arch.attn_num_heads", required=False, type=int, default=2)
    parser.add_argument("--model.arch.attn_dim", required=False, type=int, default=128)
    parser.add_argument("--model.arch.attn_drop", required=False, type=str2bool, default=False)
    parser.add_argument(
        "--model.arch.agg_fn", required=False, type=str,
        choices=["mean", "max", "attention"]
    )
    parser.add_argument("--model.arch.in_channel_2d", default=1, type=int)
    parser.add_argument(
        "--model.arch.initialization", default="custom", choices=["default", "custom"]
    )

    # training / testing related args
    parser.add_argument("--train.batch_size", required=False, type=int, default=128)
    parser.add_argument("--test.batch_size", required=False, type=int, default=128)
    # optimizer/scheduler
    parser.add_argument("--train.patience", required=False, type=int, default=20)
    parser.add_argument("--train.max_epoch", required=False, type=int, default=100)
    parser.add_argument(
        "--train.optimizer", required=False, type=str, default="adam",
        choices=["adam", "sgd"]
    )
    parser.add_argument("--train.lr", required=False, type=float, default=1e-3)
    parser.add_argument("--train.weight_decay", required=False, type=float, default=5e-4)
    parser.add_argument("--train.gradient_norm_clip", required=False, type=float, default=-1)
    parser.add_argument("--train.scheduler", required=False, type=str, default=None)
    parser.add_argument("--train.scheduler_gamma", required=False, type=float)
    parser.add_argument("--train.scheduler_milestones", required=False, nargs="+")
    parser.add_argument("--train.scheduler_patience", required=False, type=int)
    parser.add_argument("--train.scheduler_step_size", required=False, type=int)
    parser.add_argument("--train.scheduler_load_on_reduce", required=False, type=str2bool)
    parser.add_argument("--train.accumulation_steps", type=int, default=1)

    parser.add_argument(
        "--train.save_strategy", required=False, nargs="+",
        choices=["best", "last", "init", "epoch", "current"], default=["best"]
    )
    parser.add_argument("--train.log_every", required=False, type=int, default=1000)

    parser.add_argument("--train.stopping_criteria", required=False, type=str, default=None)
    parser.add_argument(
        "--train.stopping_criteria_direction", required=False, choices=["bigger", "lower"],
        default="lower"
    )
    parser.add_argument("--train.evaluations", required=False, nargs="*", choices=[])
    parser.add_argument("--test.evaluations", required=False, nargs="*", choices=[])
    parser.add_argument(
        "--test.eval_model", required=False, type=str, choices=["best", "last", "current"],
        default="best"
    )

    return parser


if __name__ == "__main__":
    # set seeds etc here
    torch.backends.cudnn.benchmark = True

    # define logger etc
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger()

    parser = parser_setup()
    config = os_utils.parse_args(parser)
    if config.seed is not None:
        os_utils.set_seed(config.seed)
    if config.debug:
        logger.setLevel(logging.DEBUG)

    logger.info("Config:")
    logger.info(pprint.pformat(config.to_dict(), indent=4))

    # see https://github.com/wandb/client/issues/714
    os_utils.safe_makedirs(config.result_folder)
    statefile, run_id, result_folder = os_utils.get_state_params(
        config.wandb.use, config.wandb.run_id, config.result_folder, config.statefile
    )
    config.statefile = statefile
    config.wandb.run_id = run_id
    config.result_folder = result_folder

    if statefile is not None:
        data = torch.load(open(statefile, "rb"), pickle_module=dill)
        epoch = data["epoch"]
        if epoch >= config.train.max_epoch:
            logger.error("Aleady trained upto max epoch; exiting")
            sys.exit()

    if config.wandb.use:
        wandb.init(
            name=config.exp_name if config.exp_name is not None else config.result_folder,
            config=config.to_dict(),
            project=config.project,
            dir=config.result_folder,
            resume=config.wandb.run_id,
            id=config.wandb.run_id,
            sync_tensorboard=True,
        )
        logger.info(f"Starting wandb with id {wandb.run.id}")

    # NOTE: WANDB creates git patch so we probably can get rid of this in future
    os_utils.copy_code("src", config.result_folder, replace=True)
    json.dump(
        config.to_dict(),
        open(f"{wandb.run.dir if config.wandb.use else config.result_folder}/config.json", "w")
    )

    logger.info("Getting data and dataloaders")
    data, meta = get_dataset(**config.data, device=config.device)

    # num_workers = max(min(os.cpu_count(), 8), 1)
    num_workers = os.cpu_count()
    logger.info(f"Using {num_workers} workers")
    train_loader = DataLoader(
        data["train"], shuffle=True, batch_size=config.train.batch_size, num_workers=num_workers
    )
    valid_loader = DataLoader(
        data["valid"], shuffle=False, batch_size=config.test.batch_size, num_workers=num_workers
    )
    test_loader = DataLoader(
        data["test"], shuffle=False, batch_size=config.test.batch_size, num_workers=num_workers
    )

    logger.info("Getting model")
    # load arch module
    arch_module = importlib.import_module(config.model.arch.file.replace("/", ".")[:-3])
    model_arch = arch_module.get_arch(
        input_shape=meta.get("input_shape"), output_size=meta.get("num_class"),
        **config.model.arch, slice_dim=config.data.frame_dim,
        in_channel=config.model.arch.in_channel_2d
    )

    # declaring models
    if config.model.name == "binary_classifier":
        from src.models.binary_classifier import BinaryClassifier

        model = BinaryClassifier(**model_arch)
    elif config.model.name == "regression":
        from src.models.regression import Regression

        model = Regression(**model_arch)
    else:
        raise Exception("Unknown model")

    model.to(config.device)
    model.stats()

    if config.wandb.use and config.wandb.watch:
        wandb.watch(model, log="all")

    # declaring trainer
    optimizer, scheduler = optimizer_utils.get_optimizer_scheduler(
        model, lr=config.train.lr,
        optimizer=config.train.optimizer,
        scheduler=config.train.get("scheduler", None),
        opt_params={
            "weight_decay": config.train.get("weight_decay", 1e-4),
            "momentum"    : config.train.get("optimizer_momentum", 0.9)
        },
        scheduler_params={
            "gamma"         : config.train.get("scheduler_gamma", 0.1),
            "milestones"    : config.train.get("scheduler_milestones", [100, 200, 300]),
            "patience"      : config.train.get("scheduler_patience", 100),
            "step_size"     : config.train.get("scheduler_step_size", 100),
            "load_on_reduce": config.train.get("scheduler_load_on_reduce"),
            "mode"          : "max" if config.train.get(
                "stopping_criteria_direction"
            ) == "bigger" else "min"
        },
    )

    logger.info("optimizer")
    logger.info(optimizer)
    logger.info("scheduler")
    logger.info(scheduler)

    if config.model.name == "binary_classifier":
        TrainerCls = BinaryClfTrainer
    else:
        TrainerCls = RegressionTrainer

    trainer = TrainerCls(
        model, optimizer, scheduler=scheduler,
        statefile=config.statefile,
        result_dir=config.result_folder,
        log_every=config.train.log_every,
        save_strategy=config.train.save_strategy,
        patience=config.train.patience,
        max_epoch=config.train.max_epoch,
        stopping_criteria=config.train.stopping_criteria,
        gradient_norm_clip=config.train.gradient_norm_clip,
        stopping_criteria_direction=config.train.stopping_criteria_direction,
        evaluations=Box(
            {
                "train": config.train.evaluations,
                "test" : config.test.evaluations
            }
        ),
        accumulation_steps=config.train.accumulation_steps
    )

    if "train" in config.mode:
        logger.info("starting training")
        trainer.train(train_loader, valid_loader)
        logger.info("Training done;")

        # copy current step and write test results to
        step_to_write = trainer.step
        step_to_write += 1

        if "test" in config.mode and config.test.eval_model == "best":
            if os.path.exists(f"{trainer.result_dir}/best_model.pt"):
                logger.info("Loading best model")
                trainer.load(f"{trainer.result_dir}/best_model.pt")
            else:
                logger.info(
                    "eval_model is best, but best model not found ::: evaluating last model"
                )
        else:
            logger.info("eval model is not best, so skipping loading at end of training")

    if "test" in config.mode:
        logger.info("evaluating model on test set")
        logger.info(f"Model was trained upto {trainer.epoch}")

        # copy current step and write test results to
        step_to_write = trainer.step
        logger.info("evaluating model on test set")
        loss, aux_loss, outputs = trainer.test(test_loader)
        logging_utils.loss_logger_helper(
            loss, aux_loss, writer=trainer.summary_writer,
            force_print=True, step=step_to_write,
            epoch=trainer.epoch,
            log_every=trainer.log_every, string="test",
            new_line=True
        )
        logging_utils.write_predictions(f"{config.result_folder}/test.csv", outputs)

        logger.info("evaluating model on train set")
        loss, aux_loss, outputs = trainer.test(train_loader)
        logging_utils.loss_logger_helper(
            loss, aux_loss, writer=trainer.summary_writer,
            force_print=True, step=step_to_write,
            epoch=trainer.epoch,
            log_every=trainer.log_every, string="train_eval",
            new_line=True
        )
        logging_utils.write_predictions(f"{config.result_folder}/train.csv", outputs)

        logger.info("evaluating model on valid set")
        loss, aux_loss, outputs = trainer.test(valid_loader)
        logging_utils.loss_logger_helper(
            loss, aux_loss, writer=trainer.summary_writer,
            force_print=True, step=step_to_write,
            epoch=trainer.epoch,
            log_every=trainer.log_every, string="valid_eval",
            new_line=True
        )
        logging_utils.write_predictions(f"{config.result_folder}/valid.csv", outputs)
