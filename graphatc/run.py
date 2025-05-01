import argparse
import json
import os
import warnings
import subprocess
import concurrent.futures

import numpy as np
import pandas as pd
import torch
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm

from graphatc.dataset import BaseDataLoader
from graphatc.dataset.uni_dataset import ATCDataset
from graphatc.dataset.common import const
from graphatc.model import MODEL_MAP
from graphatc.trainer import Trainer
from graphatc.util import str2bool

# https://github.com/pytorch/pytorch/issues/100469
warnings.filterwarnings("ignore")
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def sync_to_oss():
    command = f"ossutil sync log oss://atcdgnn/log --update --retry-times=500"
    with open("ossutil.log", "a", encoding="utf-8") as output_file:
        subprocess.call(command, shell=True, stdout=output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("low", type=int)
    parser.add_argument("high", type=int)
    parser.add_argument('--DID_LIST', nargs='*', type=str)
    parser.add_argument("--MODEL", default='DGCN', type=str, choices=MODEL_MAP.keys())
    parser.add_argument("--MODEL_TYPE", default='graph')
    parser.add_argument("--USE_BIDIRECTIONAL_RNN",
                        type=str2bool, default=False)
    parser.add_argument("--USE_MULTI_ATTN", type=str2bool, default=False)
    parser.add_argument("--MULTI_ATTN_HEAD_NUM", type=int, default=0)
    parser.add_argument("--GAT_HEAD_NUM", type=int, default=0)
    parser.add_argument("--SPLIT_COMPONENT", type=str2bool, default=False)
    parser.add_argument("--USE_SKIP_CONNECT", type=str2bool, default=True)
    parser.add_argument("--USE_SOFTMAX_AGGR", type=str2bool, default=True)
    parser.add_argument("--USE_MSG_NORM", type=str2bool, default=False)
    parser.add_argument("--POLYMER_METHOD", type=int,
                        default=const.IMPROVE_POLYMER_DELETE_STAR_CONNECT_ATOM)
    parser.add_argument('--TEXTCNN_FILTER', nargs='*', type=int)
    parser.add_argument('--AUTHOR', type=str, default=None)
    parser.add_argument('--V08303', type=str, default=None)
    parser.add_argument("--LR", default=1e-3, type=float)
    parser.add_argument("--WEIGHT_DECAY", default=1e-3, type=float)
    parser.add_argument("--NUM_LAYERS", default=11, type=int)
    parser.add_argument("--HID_DIM", default=256, type=int)
    parser.add_argument("--DROPOUT", default=0.0, type=float)
    parser.add_argument("--BATCH_SIZE", default=256, type=int)
    parser.add_argument("-e", "--EPOCHS", default=200, type=int)
    parser.add_argument("--RECORD_EPOCH", default=10,
                        type=int, choices=range(2, 30))
    parser.add_argument("--GPU", default='0', type=str)
    parser.add_argument("--LEVEL", default='1', type=str, choices=["1", "2"])
    parser.add_argument("--LOG_PATH", type=str)
    parser.add_argument("--EXP_NAME", type=str, default="TEST")
    parser.add_argument("--TRAIN_METHOD", type=str, default='Jackknife', choices=["Jackknife", "K-Fold", "all"]) # 'all' for all data training, foo validation
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--VAL_BATCH_SIZE", type=int, default=1)
    parser.add_argument("--SEED", type=int, default=42)
    parser.add_argument("--AMP", action='store_true')
    parser.add_argument("--WRITER", action='store_true')
    parser.add_argument("--OSS", type=str2bool, default=True)
    parser.add_argument("--SAVE_MODEL", action='store_true')
    args = parser.parse_args()

    dataset = ATCDataset(level=int(args.LEVEL),
                         mode=args.MODEL_TYPE,
                         polymer_method=args.POLYMER_METHOD,
                         author=args.AUTHOR,
                         cuda_idx=args.GPU,
                         split_component=args.SPLIT_COMPONENT,
                         br08303_version=args.V08303,)

    print(f"len(DATASET)={len(dataset)}")

    N_CLASS = len(dataset.get_all_atc_code_by_tree_on_level(int(args.LEVEL)))
    N_SMILES_VOCAB = len(dataset.smiles_vocab_map) if args.MODEL_TYPE == "smiles" else 0
    
    if args.LEVEL == "1":
        assert N_CLASS == 14
    elif args.LEVEL == "2":
        assert N_CLASS == 93
    else:
        pass  # unreachable

    train_args = {'batch_size': args.BATCH_SIZE, 'drop_last': False, 'shuffle': False,
                  'num_workers': 0,
                  'prefetch_factor': None
                  }

    data_loader = BaseDataLoader(
        dataset, GraphDataLoader, train_args, val_batch_size=args.VAL_BATCH_SIZE, seed=args.SEED)

    LOG_PATH = f'./log/{args.EXP_NAME}_L{args.LEVEL}/'
    DEVICE = 'cuda:' + args.GPU
    if args.DID_LIST is None:
        LOW_HIGH_RANGE = range(args.low, args.high)
    else:
        LOW_HIGH_RANGE = [dataset.did_to_index(x) for x in args.DID_LIST]

    if args.TRAIN_METHOD == "Jackknife":
        LOG_PATH += "[DID]" if args.DID_LIST is not None else ""
        LOG_PATH += f'[{LOW_HIGH_RANGE[0]:04}-{LOW_HIGH_RANGE[-1]:04}]'
    elif args.TRAIN_METHOD == "K-Fold":
        LOG_PATH += f'[{args.K}-Fold][{LOW_HIGH_RANGE[0]:03}-{LOW_HIGH_RANGE[-1]:03}]'
    elif args.TRAIN_METHOD == "all":
        LOG_PATH += f'[all][{LOW_HIGH_RANGE[0]:03}-{LOW_HIGH_RANGE[-1]:03}]'

    LOG_PATH += f'N{args.NUM_LAYERS}H{args.HID_DIM}_' \
                f'BR={args.USE_BIDIRECTIONAL_RNN}_' \
                f'MA={args.USE_MULTI_ATTN}_HEAD{args.MULTI_ATTN_HEAD_NUM}_' \
                f'SKIP={args.USE_SKIP_CONNECT}_AGGR={args.USE_SOFTMAX_AGGR}_MSGNORM={args.USE_MSG_NORM}_' \
                f'LR{args.LR:.0e}_' \
                f'WD{args.WEIGHT_DECAY:.0e}_' \
                f'B{args.BATCH_SIZE}_e{args.EPOCHS}_' \
                f'P{args.POLYMER_METHOD}_' \
                f'AMP={args.AMP}'

    if not args.LOG_PATH:
        args.LOG_PATH = LOG_PATH

    if not os.path.exists(args.LOG_PATH):
        os.makedirs(args.LOG_PATH)

    print(args)

    with open(LOG_PATH + '/args.json', 'w', encoding="utf-8") as f:
        json.dump(vars(args), f)

    if args.TRAIN_METHOD == "Jackknife":
        fold_bar = tqdm(data_loader.get_data_loader(args.TRAIN_METHOD, val_indices_range=LOW_HIGH_RANGE), total=len(LOW_HIGH_RANGE),
                        desc="[JACKN]", unit='fold', position=1, leave=False, colour='red', ncols=60)
    elif args.TRAIN_METHOD == "K-Fold" or args.TRAIN_METHOD == "all":
        fold_bar = tqdm(data_loader.get_data_loader(args.TRAIN_METHOD, K=args.K, K_range=LOW_HIGH_RANGE), total=len(LOW_HIGH_RANGE),
                        desc="[KFOLD]", unit='fold', position=1, leave=False, colour='red', ncols=60)

    val_predict_opt_jn = pd.DataFrame()
    aiming_opt = 0 # save the best aiming
    for k, (train_loader, val_loader) in enumerate(fold_bar):
        model = MODEL_MAP[args.MODEL](
            hid_dim=args.HID_DIM,
            out_dim=N_CLASS,
            num_layers=args.NUM_LAYERS,
            dropout=args.DROPOUT,
            num_node_emb_list=[120, 4],
            num_edge_emb_list=[4 + 1, 5],  # +1 for self loop
            use_bidirectional_rnn=args.USE_BIDIRECTIONAL_RNN,
            use_multi_attn=args.USE_MULTI_ATTN,
            multi_attn_head_num=args.MULTI_ATTN_HEAD_NUM,
            use_skip_connect=args.USE_SKIP_CONNECT,
            use_softmax_aggr=args.USE_SOFTMAX_AGGR,
            use_msgnorm=args.USE_MSG_NORM,
            n_vocab=N_SMILES_VOCAB,
            filter_sizes=args.TEXTCNN_FILTER,
            gat_num_heads=args.GAT_HEAD_NUM,

        ).to(DEVICE)

        if args.TRAIN_METHOD == "Jackknife":
            _val_index = list(LOW_HIGH_RANGE[k:k+1])
        elif args.TRAIN_METHOD == "K-Fold" or args.TRAIN_METHOD == "all":
            _val_index = val_loader.sampler.indices

        trainer = Trainer(model, train_loader, val_loader,
                          criterion=torch.nn.MultiLabelSoftMarginLoss(),
                          optimizer=torch.optim.AdamW(
                              model.parameters(), lr=args.LR, weight_decay=args.WEIGHT_DECAY),
                          epochs=args.EPOCHS,
                          record_epoch=args.RECORD_EPOCH,
                          log_path=args.LOG_PATH,
                          val_index=_val_index,
                          writer=args.WRITER,
                          level=args.LEVEL,
                          use_amp=args.AMP,
                          n_class=N_CLASS,
                          model_type=args.MODEL_TYPE,
                          input_args=vars(args)
                          )

        val_predict_opt = trainer.train()
        val_predict_opt_jn = pd.concat([val_predict_opt_jn, val_predict_opt])
        val_predict_opt_jn.to_csv(args.LOG_PATH + '/predict.csv')
        if args.OSS:
            executor.submit(sync_to_oss)

        if args.SAVE_MODEL:
            curr_aiming = val_predict_opt[f"aiming_l{int(args.LEVEL)}"].values.mean()
            if curr_aiming > aiming_opt:
                aiming_opt = curr_aiming
                torch.save({'model': model.state_dict()},
                        args.LOG_PATH + '/model_state_dict.pth')

    print(args)
    print(val_predict_opt_jn[[f'aiming_l{args.LEVEL}']].mean())
    print(val_predict_opt_jn[[f'coverage_l{args.LEVEL}']].mean())
    print(val_predict_opt_jn[[f'accuracy_l{args.LEVEL}']].mean())
    print(val_predict_opt_jn[[f'abs_true_l{args.LEVEL}']].mean())
    print(val_predict_opt_jn[[f'abs_false_l{args.LEVEL}']].mean())
