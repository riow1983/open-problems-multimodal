# -*- coding: utf-8 -*-



"""# CFG"""

# ====================================================
# CFG
# ====================================================
import yaml
import io

class CFG:
    def __init__(self, filename):
        with io.open(filename, 'r') as stream:
            cfg = yaml.safe_load(stream)
        
        self.debug = cfg['debug']
        
        self.exp = cfg['type'] +'-'+ cfg['exp']
        self.comp_name = cfg['comp_name']
        self.nb_name = cfg['nb_name']
        self.dname = f'kaggle001-{cfg["type"]}-cv'
        self.cv_x = f'X_{cfg["type"]}_fold.h5'
        self.cv_y = f'Y_{cfg["type"]}_fold.h5'
        self.wandb = cfg['wandb']
        self.wandbgroup = cfg['nb_name']
        self.wandbproject = cfg['comp_name']
        self.wandbname = self.exp
        self.competition = cfg['comp_name']
        self._wandb_kernel = cfg['_wandb_kernel']
        self.apex = cfg['apex']
        self.print_freq = cfg['print_freq']
        self.num_workers = cfg['num_workers']
        self.scheduler = cfg['scheduler']
        self.batch_scheduler = cfg['batch_scheduler']
        self.num_cycles = cfg['num_cycles']
        self.num_warmup_steps = cfg['num_warmup_steps']
        self.epochs = cfg['epochs']
        self.encoder_lr = cfg['encoder_lr']
        self.decoder_lr = cfg['decoder_lr']
        self.min_lr = cfg['min_lr']
        self.eps = cfg['eps']
        self.betas = tuple(cfg['betas'])
        self.batch_size = cfg['batch_size']
        self.fc_dropout = cfg['fc_dropout']
        self.weight_decay = cfg['weight_decay']
        self.gradient_accumulation_steps = cfg['gradient_accumulation_steps']
        self.max_grad_norm = cfg['max_grad_norm']
        self.seed = cfg['seed']
        self.n_fold = cfg['n_fold']
        self.trn_fold = cfg['trn_fold']
        self.train = cfg['train']
    
        if self.debug:
            self.epochs = 2
            self.trn_fold = [0]
            self.wandbname = "debug-" + self.wandbname


args = CFG('./config.yaml')





# Commented out IPython magic to ensure Python compatibility.
# ====================================================
# Directory settings
# ====================================================

import sys
import os
from pathlib import Path

KAGGLE_ENV = True if 'KAGGLE_URL_BASE' in set(os.environ.keys()) else False

if KAGGLE_ENV:
    BASE_DIR = Path('/kaggle/working')
elif "google.colab" in sys.modules:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = Path(f"/content/drive/MyDrive/colab_notebooks/kaggle/{args.comp_name}")
    WANDB_PATH = Path("/content/drive/MyDrive/colab_notebooks/kaggle/wandb.json")
    LINE_PATH = Path("/content/drive/MyDrive/colab_notebooks/kaggle/line.json")
else:
    BASE_DIR = Path("/home/jovyan/kaggle")
    WANDB_PATH = Path("/home/jovyan/kaggle/wandb.json")
    LINE_PATH = Path("/home/jovyan/kaggle/line.json")


INPUT_DIR = BASE_DIR / 'input'
#!mkdir {INPUT_DIR}
os.makedirs(INPUT_DIR)

if KAGGLE_ENV:
    OUTPUT_DIR = Path('')
else:
    OUTPUT_DIR = INPUT_DIR / args.nb_name
    #!mkdir {OUTPUT_DIR}
    os.makedirs(OUTPUT_DIR)




# ====================================================
# wandb
# ====================================================
if args.wandb:
    # if 'google.colab' in sys.modules:
    #     !pip install wandb
    os.system('pip install wandb')
    import wandb

    try:
        if 'kaggle_web_client' in sys.modules:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            secret_value_0 = user_secrets.get_secret("wandb_api")
        else:
            import json
            f = open(WANDB_PATH, "r")
            json_data = json.load(f)
            secret_value_0 = json_data["wandb_api"]
        wandb.login(key=secret_value_0)
        anony = None
    except:
        anony = "must"
        print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')


    def class2dict(f):
        return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

    run = wandb.init(dir=OUTPUT_DIR,
                     project=args.wandbproject, 
                     name=args.wandbname,
                     config=class2dict(args),
                     group=args.wandbgroup,
                     job_type="train",
                     anonymous=anony)
    print(f"wandb run id: {run.id}")

"""# Library"""

# Commented out IPython magic to ensure Python compatibility.
# ====================================================
# Library
# ====================================================
import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
# from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

from model import *
from dataset import *
from loss import *



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""# Helper functions for scoring"""

def correlation_score(y_true, y_pred):
    """Scores the predictions according to the competition rules. 
    
    It is assumed that the predictions are not constant.
    
    Returns the average of each sample's Pearson correlation coefficient"""
    if type(y_true) == pd.DataFrame: y_true = y_true.values
    if type(y_pred) == pd.DataFrame: y_pred = y_pred.values
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)




# def micro_f1(preds, truths):
#     """
#     Micro f1 on binary arrays.
#     Args:
#         preds (list of lists of ints): Predictions.
#         truths (list of lists of ints): Ground truths.
#     Returns:
#         float: f1 score.
#     """
#     # Micro : aggregating over all instances
#     preds = np.concatenate(preds)
#     truths = np.concatenate(truths)
#     return f1_score(truths, preds)


# def spans_to_binary(spans, length=None):
#     """
#     Converts spans to a binary array indicating whether each character is in the span.
#     Args:
#         spans (list of lists of two ints): Spans.
#     Returns:
#         np array [length]: Binarized spans.
#     """
#     length = np.max(spans) if length is None else length
#     binary = np.zeros(length)
#     for start, end in spans:
#         binary[start:end] = 1
#     return binary


# def span_micro_f1(preds, truths):
#     """
#     Micro f1 on spans.
#     Args:
#         preds (list of lists of two ints): Prediction spans.
#         truths (list of lists of two ints): Ground truth spans.
#     Returns:
#         float: f1 score.
#     """
#     bin_preds = []
#     bin_truths = []
#     for pred, truth in zip(preds, truths):
#         if not len(pred) and not len(truth):
#             continue
#         length = max(np.max(pred) if len(pred) else 0, np.max(truth) if len(truth) else 0)
#         bin_preds.append(spans_to_binary(pred, length))
#         bin_truths.append(spans_to_binary(truth, length))
#     return micro_f1(bin_preds, bin_truths)

# def create_labels_for_scoring(df):
#     # example: ['0 1', '3 4'] -> ['0 1; 3 4']
#     df['location_for_create_labels'] = [ast.literal_eval(f'[]')] * len(df)
#     for i in range(len(df)):
#         lst = df.loc[i, 'location']
#         if lst:
#             new_lst = ';'.join(lst)
#             df.loc[i, 'location_for_create_labels'] = ast.literal_eval(f'[["{new_lst}"]]')
#     # create labels
#     truths = []
#     for location_list in df['location_for_create_labels'].values:
#         truth = []
#         if len(location_list) > 0:
#             location = location_list[0]
#             for loc in [s.split() for s in location.split(';')]:
#                 start, end = int(loc[0]), int(loc[1])
#                 truth.append([start, end])
#         truths.append(truth)
#     return truths


# def get_char_probs(texts, predictions, tokenizer):
#     results = [np.zeros(len(t)) for t in texts]
#     for i, (text, prediction) in enumerate(zip(texts, predictions)):
#         encoded = tokenizer(text, 
#                             add_special_tokens=True,
#                             return_offsets_mapping=True)
#         for idx, (offset_mapping, pred) in enumerate(zip(encoded['offset_mapping'], prediction)):
#             start = offset_mapping[0]
#             end = offset_mapping[1]
#             results[i][start:end] = pred
#     return results


# def get_results(char_probs, th=0.5):
#     results = []
#     for char_prob in char_probs:
#         result = np.where(char_prob >= th)[0] + 1
#         result = [list(g) for _, g in itertools.groupby(result, key=lambda n, c=itertools.count(): n - next(c))]
#         result = [f"{min(r)} {max(r)}" for r in result]
#         result = ";".join(result)
#         results.append(result)
#     return results


# def get_predictions(results):
#     predictions = []
#     for result in results:
#         prediction = []
#         if result != "":
#             for loc in [s.split() for s in result.split(';')]:
#                 start, end = int(loc[0]), int(loc[1])
#                 prediction.append([start, end])
#         predictions.append(prediction)
#     return predictions

"""# Utils"""

# ====================================================
# Utils
# ====================================================
def get_score(y_true, y_pred):
    score = correlation_score(y_true, y_pred)
    return score


def get_logger(filename=OUTPUT_DIR+'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=args.seed)

"""# Data Loading"""

# ====================================================
# Data Loading
# ====================================================

FP_CELL_METADATA = INPUT_DIR / "metadata.csv"

FP_TRAIN_INPUTS = INPUT_DIR / f"train_{args.type}_inputs.h5"
FP_TRAIN_TARGETS = INPUT_DIR / f"train_{args.type}_targets.h5"
FP_TEST_INPUTS = INPUT_DIR / f"test_{args.type}_inputs.h5"

FP_SUBMISSION = INPUT_DIR / "sample_submission.csv"
FP_EVALUATION_IDS = INPUT_DIR / "evaluation_ids.csv"

train = pd.read_hdf(FP_TRAIN_INPUTS)


print(f"train.shape: {train.shape}")
#display(train.head())






"""# CV split"""

# ====================================================
# CV split
# ====================================================
# Fold = GroupKFold(n_splits=CFG.n_fold)
# groups = train['pn_num'].values
# for n, (train_index, val_index) in enumerate(Fold.split(train, train['location'], groups)):
#     train.loc[val_index, 'fold'] = int(n)
# train['fold'] = train['fold'].astype(int)
#display(train.groupby('fold').size())
X_train = pd.read_hdf(INPUT_DIR / args.dname / args.cv_x)
Y_train = pd.read_hdf(INPUT_DIR / args.dname / args.cv_y)

if args.debug:
    #display(train.groupby('fold').size())
    if len(train) > 2000:
        train = train.sample(n=2000, random_state=0).reset_index(drop=True)
        #display(train.groupby('fold').size())



"""# Dataset"""

# ====================================================
# Dataset
# ====================================================



"""# Model"""

# ====================================================
# Model
# ====================================================



"""# Helper functions"""

# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=args.apex):
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, args.num_tasks), labels.view(-1, args.num_tasks))
        #loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if (step + 1) % args.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if args.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % args.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))
        if args.wandb:
            wandb.log({f"[fold{fold}] loss": losses.val,
                       f"[fold{fold}] lr": scheduler.get_lr()[0]})
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, args.num_tasks), labels.view(-1, args.num_tasks))
        # loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        # preds.append(y_preds.sigmoid().to('cpu').numpy())
        preds.append(y_preds.to('cpu').numpy())
        end = time.time()
        if step % args.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
    predictions = np.concatenate(preds)
    return losses.avg, predictions


def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        # for k, v in inputs.items():
        #     inputs[k] = v.to(device)
        inputs = inputs.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        # preds.append(y_preds.sigmoid().to('cpu').numpy())
        preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions

# ====================================================
# train loop
# ====================================================
def train_loop(X_folds, Y_folds, fold):
    
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    # train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    # valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    X_train_folds = X_folds[X_folds['fold'] != fold].reset_index(drop=True)
    Y_train_folds = Y_folds[Y_folds['fold'] != fold].reset_index(drop=True)
    X_valid_folds = X_folds[X_folds['fold'] == fold].reset_index(drop=True)
    Y_valid_folds = Y_folds[Y_folds['fold'] == fold].reset_index(drop=True)
    # valid_texts = valid_folds['pn_history'].values
    # valid_labels = create_labels_for_scoring(valid_folds)
    
    train_dataset = TrainDataset(X_train_folds, Y_train_folds)
    valid_dataset = TrainDataset(X_valid_folds, Y_valid_folds)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = MultipleRegression(args)
    torch.save(model.config, OUTPUT_DIR / 'config.pth')
    model.to(device)
    
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=args.encoder_lr, 
                                                decoder_lr=args.decoder_lr,
                                                weight_decay=args.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=args.encoder_lr, eps=args.eps, betas=args.betas)
    
    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler=='linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler=='cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
            )
        return scheduler
    
    num_train_steps = int(len(Y_train_folds) / args.batch_size * args.epochs)
    scheduler = get_scheduler(args, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    # criterion = nn.BCEWithLogitsLoss(reduction="none")
    criterion = NegativeCorrLoss()
    
    best_score = 0.

    for epoch in range(args.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device)
        # predictions = predictions.reshape((len(valid_folds), args.max_len))
        
        # scoring
        # char_probs = get_char_probs(valid_texts, predictions, args.tokenizer)
        # results = get_results(char_probs, th=0.5)
        # preds = get_predictions(results)
        # score = get_score(valid_labels, preds)
        score = get_score(Y_valid_folds, preds)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')
        if args.wandb:
            wandb.log({f"[fold{fold}] epoch": epoch+1, 
                       f"[fold{fold}] avg_train_loss": avg_loss, 
                       f"[fold{fold}] avg_val_loss": avg_val_loss,
                       f"[fold{fold}] score": score})
        
        if best_score < score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                        OUTPUT_DIR / f"{args.model.replace('/', '-')}_fold{fold}_best.pth")

    predictions = torch.load(OUTPUT_DIR / f"{args.model.replace('/', '-')}_fold{fold}_best.pth", 
                             map_location=torch.device('cpu'))['predictions']
    # valid_folds[[i for i in range(args.max_len)]] = predictions

    torch.cuda.empty_cache()
    gc.collect()
    
    # return valid_folds
    return pd.DataFrame(predictions)

if __name__ == '__main__':
    
    # def get_result(oof_df, cv_score=False, case_num=None):
    #     if case_num is not None:
    #         oof_df = oof_df[oof_df["case_num"]==case_num].reset_index(drop=True)
    #     labels = create_labels_for_scoring(oof_df)
    #     predictions = oof_df[[i for i in range(args.max_len)]].values
    #     char_probs = get_char_probs(oof_df['pn_history'].values, predictions, CFG.tokenizer)
    #     results = get_results(char_probs, th=0.5)
    #     preds = get_predictions(results)
    #     score = get_score(labels, preds)
 
    #     LOGGER.info(f'Score: {score:<.4f}')
    #     if cv_score:
    #         wandb.log({f'CV score': score})

    def get_result(oof_df, Y, cv_score=False):
        # labels = create_labels_for_scoring(oof_df)
        labels = Y.values[:, :-1]
        predictions = oof_df.values
        score = get_score(labels, predictions)

        LOGGER.info(f'Score: {score:<.4f}')
        if cv_score:
            wandb.log({f'CV score': score})
    
    if args.train:
        oof_df = pd.DataFrame()
        for fold in range(args.n_fold):
            if fold in args.trn_fold:
                _oof_df = train_loop(X_train, Y_train, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df, Y_train)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")

        get_result(oof_df, Y_train, cv_score=True)
        oof_df.to_pickle(OUTPUT_DIR / 'oof_df.pkl')
        
    if args.wandb:
        wandb.finish()


    # Push to LINE
    import requests
    def send_line_notification(message):
        import json
        f = open("../../line.json", "r")
        json_data = json.load(f)
        line_token = json_data["kagglePush"]
        endpoint = 'https://notify-api.line.me/api/notify'
        message = "\n{}".format(message)
        payload = {'message': message}
        headers = {'Authorization': 'Bearer {}'.format(line_token)}
        requests.post(endpoint, data=payload, headers=headers)

    if args.wandb:
        send_line_notification(f"Training of {args.wandbgroup} has been done. See {run.url}")
    else:
        send_line_notification(f"Training of {args.wandbgroup} has been done.")

