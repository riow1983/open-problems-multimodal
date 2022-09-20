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
        
        self.comp_name = cfg['comp_name']
        self.nb_name = cfg['nb_name']
        self.wandb = cfg['wandb']
        self.wandbgroup = cfg['nb_name']
        self.wandbproject = cfg['comp_name']
        self.wandbname = cfg['exp']
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


args = CFG()





# Commented out IPython magic to ensure Python compatibility.
# ====================================================
# Directory settings
# ====================================================
import os
import sys
if "google.colab" in sys.modules:
    from google.colab import drive
    drive.mount("/content/drive")
    base = f"/content/drive/MyDrive/colab_notebooks/kaggle/{args.nb_name}/src"
    os.chdir(base)

# if 'kaggle_web_client' in sys.modules:
#     OUTPUT_DIR = './'
# else:
#     OUTPUT_DIR = './001t-token-classifier/'
# if not os.path.exists(OUTPUT_DIR):
#     os.makedirs(OUTPUT_DIR)


KAGGLE_ENV = True if 'KAGGLE_URL_BASE' in set(os.environ.keys()) else False
INPUT_DIR = Path('../input/')

if KAGGLE_ENV:
    OUTPUT_DIR = Path('')
else:
    os.makedirs(f'../input/{args.nb_name}')
    OUTPUT_DIR = INPUT_DIR / args.nb_name




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
            f = open("../../wandb.json", "r")
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
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""# Helper functions for scoring"""


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
    score = span_micro_f1(y_true, y_pred)
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
    
seed_everything(seed=42)

"""# Data Loading"""

# ====================================================
# Data Loading
# ====================================================
train = pd.read_csv('../input/nbme-score-clinical-patient-notes/train.csv')
train['annotation'] = train['annotation'].apply(ast.literal_eval)
train['location'] = train['location'].apply(ast.literal_eval)
features = pd.read_csv('../input/nbme-score-clinical-patient-notes/features.csv')
def preprocess_features(features):
    features.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"
    return features
features = preprocess_features(features)
patient_notes = pd.read_csv('../input/nbme-score-clinical-patient-notes/patient_notes.csv')

print(f"train.shape: {train.shape}")
#display(train.head())
print(f"features.shape: {features.shape}")
#display(features.head())
print(f"patient_notes.shape: {patient_notes.shape}")
#display(patient_notes.head())


"""## Merge patient_notes w/ features"""

print(patient_notes.shape)
patient_notes = patient_notes.merge(features, on=['case_num'], how='left')
print(patient_notes.shape)
#display(patient_notes.head())

"""## ~~Remove pn_nums which are appeared in train from patient_notes~~"""

# print(patient_notes.shape)
# patient_notes = patient_notes[~patient_notes["pn_num"].isin(train["pn_num"].unique())].reset_index(drop=True)
# print(patient_notes.shape)

"""## Select one specific case_num"""

if CFG.wandbname.split("-")[-1] != "all":
    selected_case_num = int(CFG.wandbname.split("-")[-1])
    print(f"selected_case_num: {selected_case_num}")

    print(train.shape)
    train = train[train["case_num"]==selected_case_num].reset_index(drop=True)
    print(train.shape)

    print()

    print(patient_notes.shape)
    patient_notes = patient_notes[patient_notes["case_num"]==selected_case_num].reset_index(drop=True)
    print(patient_notes.shape)

    print()

    print(features.shape)
    features = features[features["case_num"]==selected_case_num].reset_index(drop=True)
    print(features.shape)

"""# CV split"""

# ====================================================
# CV split
# ====================================================
Fold = GroupKFold(n_splits=CFG.n_fold)
groups = train['pn_num'].values
for n, (train_index, val_index) in enumerate(Fold.split(train, train['location'], groups)):
    train.loc[val_index, 'fold'] = int(n)
train['fold'] = train['fold'].astype(int)
#display(train.groupby('fold').size())

len(train)

if CFG.debug:
    #display(train.groupby('fold').size())
    if len(train) > 2000:
        train = train.sample(n=2000, random_state=0).reset_index(drop=True)
        #display(train.groupby('fold').size())

"""# tokenizer"""

# ====================================================
# tokenizer
# ====================================================
tokenizer = AutoTokenizer.from_pretrained(CFG.model)
tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
CFG.tokenizer = tokenizer

"""# Dataset"""

# ====================================================
# Define max_len
# ====================================================
for text_col in ['pn_history']:
    pn_history_lengths = []
    tk0 = tqdm(patient_notes[text_col].fillna("").values, total=len(patient_notes))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        pn_history_lengths.append(length)
    LOGGER.info(f'{text_col} max(lengths): {max(pn_history_lengths)}')

for text_col in ['feature_text']:
    features_lengths = []
    tk0 = tqdm(features[text_col].fillna("").values, total=len(features))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        features_lengths.append(length)
    LOGGER.info(f'{text_col} max(lengths): {max(features_lengths)}')

CFG.max_len = max(pn_history_lengths) + max(features_lengths) + 3 # cls & sep & sep
LOGGER.info(f"max_len: {CFG.max_len}")

# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, text, feature_text):
    inputs = cfg.tokenizer(text, feature_text, 
                           add_special_tokens=True,
                           max_length=CFG.max_len,
                           padding="max_length",
                           return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


def create_label(cfg, text, annotation_length, location_list):
    encoded = cfg.tokenizer(text,
                            add_special_tokens=True,
                            max_length=CFG.max_len,
                            padding="max_length",
                            return_offsets_mapping=True)
    offset_mapping = encoded['offset_mapping']
    ignore_idxes = np.where(np.array(encoded.sequence_ids()) != 0)[0]
    label = np.zeros(len(offset_mapping))
    label[ignore_idxes] = -1
    if annotation_length != 0:
        for location in location_list:
            for loc in [s.split() for s in location.split(';')]:
                start_idx = -1
                end_idx = -1
                start, end = int(loc[0]), int(loc[1])
                for idx in range(len(offset_mapping)):
                    if (start_idx == -1) & (start < offset_mapping[idx][0]):
                        start_idx = idx - 1
                    if (end_idx == -1) & (end <= offset_mapping[idx][1]):
                        end_idx = idx + 1
                if start_idx == -1:
                    start_idx = end_idx
                if (start_idx != -1) & (end_idx != -1):
                    label[start_idx:end_idx] = 1
    return torch.tensor(label, dtype=torch.float)


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.feature_texts = df['feature_text'].values
        self.pn_historys = df['pn_history'].values
        self.annotation_lengths = df['annotation_length'].values
        self.locations = df['location'].values

    def __len__(self):
        return len(self.feature_texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, 
                               self.pn_historys[item], 
                               self.feature_texts[item])
        label = create_label(self.cfg, 
                             self.pn_historys[item], 
                             self.annotation_lengths[item], 
                             self.locations[item])
        return inputs, label

"""# Model"""

# ====================================================
# Model
# ====================================================
class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output

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
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (inputs, labels) in enumerate(train_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
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
        if CFG.wandb:
            wandb.log({f"[fold{fold}] loss": losses.val,
                       f"[fold{fold}] lr": scheduler.get_lr()[0]})
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.sigmoid().to('cpu').numpy())
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
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
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions

# ====================================================
# train loop
# ====================================================
def train_loop(folds, fold):
    
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_texts = valid_folds['pn_history'].values
    valid_labels = create_labels_for_scoring(valid_folds)
    
    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, OUTPUT_DIR+'config.pth')
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
                                                encoder_lr=CFG.encoder_lr, 
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)
    
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
    
    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    
    best_score = 0.

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device)
        predictions = predictions.reshape((len(valid_folds), CFG.max_len))
        
        # scoring
        char_probs = get_char_probs(valid_texts, predictions, CFG.tokenizer)
        results = get_results(char_probs, th=0.5)
        preds = get_predictions(results)
        score = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')
        if CFG.wandb:
            wandb.log({f"[fold{fold}] epoch": epoch+1, 
                       f"[fold{fold}] avg_train_loss": avg_loss, 
                       f"[fold{fold}] avg_val_loss": avg_val_loss,
                       f"[fold{fold}] score": score})
        
        if best_score < score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                        OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")

    predictions = torch.load(OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth", 
                             map_location=torch.device('cpu'))['predictions']
    valid_folds[[i for i in range(CFG.max_len)]] = predictions

    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds

if __name__ == '__main__':
    
    def get_result(oof_df, cv_score=False, case_num=None):
        if case_num is not None:
            oof_df = oof_df[oof_df["case_num"]==case_num].reset_index(drop=True)
        labels = create_labels_for_scoring(oof_df)
        predictions = oof_df[[i for i in range(CFG.max_len)]].values
        char_probs = get_char_probs(oof_df['pn_history'].values, predictions, CFG.tokenizer)
        results = get_results(char_probs, th=0.5)
        preds = get_predictions(results)
        score = get_score(labels, preds)
        if case_num is not None:
            LOGGER.info(f'Score of case_num {case_num}: {score:<.4f}')
            if cv_score:
                wandb.log({f'CV score of case_num {case_num}': score})
        else:
            LOGGER.info(f'Score: {score:<.4f}')
            if cv_score:
                wandb.log({f'CV score': score})
    
    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(train, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        if CFG.cv_case_num:
            for i in range(10):
                get_result(oof_df, cv_score=True, case_num=i)
        else:
            get_result(oof_df, cv_score=True)
        oof_df.to_pickle(OUTPUT_DIR+'oof_df.pkl')
        
    if CFG.wandb:
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

    if CFG.wandb:
        send_line_notification(f"Training of {CFG.wandbgroup} has been done. See {run.url}")
    else:
        send_line_notification(f"Training of {CFG.wandbgroup} has been done.")

