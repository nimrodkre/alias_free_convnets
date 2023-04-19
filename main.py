# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# convnext_LICENSE.txt file in the root directory of this source tree.


import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torchvision
import json
import os

from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, LayerDecayValueAssigner

from datasets import build_dataset
from engine import train_one_epoch, evaluate, evaluate_consistency_metrics, evaluate_benchmark

from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from utils import get_args, get_model


from shift_eval import evaluate_fraction_adversarial_shift_metrics, evaluate_full_adversarial_shift_metrics, \
    evaluate_adversarial_crop_shift_metrics, evaluate_random_shift_metrics
try:
    import wandb

except ImportError:
    wandb = None
    raise ImportError(
        "To use the Weights and Biases Logger please install wandb."
        "Run `pip install wandb` to install it."
    )


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    print("build datasets:")
    dataset_train, args.nb_classes = build_dataset(is_train=False, args=args)
    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    if global_rank == 0 and args.enable_wandb:
        # added to support wandb sweep
        run = wandb.init(
                project=args.project,
                config=args,
                name=args.wandb_name,
                group=args.wandb_group
            )
        args = wandb.config
        print("wandb.config args: ", args)
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None


    print(f"dataset_train: \n{dataset_train}\n dataset_val: \n{dataset_val}\n nb_classes: {args.nb_classes}")

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)


    model = get_model(args)

    # sample inputs
    # matplotlib.use('Agg')  # turn off gui - use on dgx cluster
    # for loader, name in [(data_loader_train, 'train'), (data_loader_val, 'val')]:
    #     sample = next(iter(loader))[0]
    #     sample = sample[:12] if sample.shape[0] > 12 else sample
    #     sample = sample.permute(0, 2, 3, 1).cpu().numpy()
    #     fig, axs = plt.subplots(4, 3, figsize=(12, 9))
    #     axs = axs.flatten()
    #     for img, ax in zip(sample, axs):
    #         ax.imshow(img)
    #     fig.suptitle(f'{name} sample')
    #     if args.output_dir:
    #         plt.savefig(os.path.join(args.output_dir, f'{name}_sample.png'))
    #     else:
    #         plt.show()

    # Profiler analysis
    if args.profiler:
        from utils import profiler_test
        print("run profiler")
        profiler_test(model, args)

    # Wandb watch model
    if global_rank == 0 and args.enable_wandb and wandb is not None:
        wandb.watch(model, criterion=None, log="all", log_freq=1000, idx=None, log_graph=False)

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)



    if args.layer_decay < 1.0 or args.layer_decay > 1.0:
        num_layers = 12 # convnext layers divided into 12 parts, each with a different decayed lr value.
        assert args.model in ['convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'], \
             "Layer Decay impl only supports convnext_small/base/large/xlarge"
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None)

    loss_scaler = NativeScaler() # if args.use_amp is False, this won't be used

    lr_schedule_values = utils.get_lr_scheduler(num_training_steps_per_epoch, args)

    if args.weight_decay_end is None:
        # solve wandb bug
        try:
            args.update({'weight_decay_end': args.weight_decay}, allow_val_change=True)
        except:
            args.weight_decay_end = args.weight_decay


    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    # print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        print(f"Eval only mode")
        test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
   

if __name__ == '__main__':
    args = get_args()
    main(args)
