import argparse
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
import os
import time
from sklearn.model_selection import *
from transformers import *

CFG = {  # 训练的参数配置
    'fold_num': 5,  # 五折交叉验证
    'seed': 42,
    'model': 'hfl/chinese-macbert-large',  # 预训练模型
    'max_len': 512,  # 文本截断的最大长度
    'epochs': 16,
    'train_bs': 2,  # batch_size，可根据自己的显存调整
    'valid_bs': 2,
    'lr': 2e-5,  # 学习率
    'num_workers': 16,
    'accum_iter': 2,  # 梯度累积，相当于将batch_size*2
    'weight_decay': 1e-4,  # 权重衰减，防止过拟合
    'device': 0,
    'adv_lr': 0.01,
    'adv_norm_type': 'l2',
    'adv_init_mag': 0.03,
    'adv_max_norm': 1.0,
    'ip': 2,
    'gpuNum': 4,

}
parser = argparse.ArgumentParser()
parser.add_argument("-input", type=str, required=True)
parser.add_argument("-output", type=str, required=True)
args = parser.parse_args()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG['seed'])  # 固定随机种子

# torch.cuda.set_device(CFG['device'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_df = pd.read_csv(args.input)
tokenizer = BertTokenizer.from_pretrained(CFG['model'])  # 加载bert的分词器


class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):  # 将一条数据从(文章,问题,4个选项)转成(文章,问题,选项1)、(文章,问题,选项2)...
        label = self.df.label.values[idx]
        question = self.df.Question.values[idx]
        content = self.df.Content.values[idx]
        choice = self.df.Choices.values[idx][2:-2].split('\', \'')
        if len(choice) < 4:  # 如果选项不满四个，就补“不知道”
            for i in range(4 - len(choice)):
                choice.append('D．不知道')

        content = [content for i in range(len(choice))]
        pair = [question + ' ' + i[2:] for i in choice]

        return content, pair, label


def collate_fn(data):  # 将文章问题选项拼在一起后，得到分词后的数字id，输出的size是(batch, n_choices, max_len)
    input_ids, attention_mask, token_type_ids = [], [], []
    for x in data:
        text = tokenizer(x[1], text_pair=x[0], padding='max_length', truncation=True, max_length=CFG['max_len'],
                         return_tensors='pt')
        input_ids.append(text['input_ids'].tolist())
        attention_mask.append(text['attention_mask'].tolist())
        token_type_ids.append(text['token_type_ids'].tolist())
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    label = torch.tensor([x[-1] for x in data])
    return input_ids, attention_mask, token_type_ids, label

test_set = MyDataset(test_df)
test_loader = DataLoader(test_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False,
                         num_workers=CFG['num_workers'])
model = BertForMultipleChoice.from_pretrained(CFG['model']).to(device)
model = nn.DataParallel(model)

predictions = []

for fold in range(int(CFG['fold_num'])):  # 把训练后的五个模型挨个进行预测
    if fold == 4:
        continue
    y_pred = []
    model.load_state_dict(
        torch.load('LargeMac_{}_fold_{}.pt'.format(CFG['model'].split('/')[1], fold + 1), map_location=device))

    with torch.no_grad():
        tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
        for idx, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
            input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
                device), token_type_ids.to(device), y.to(device).long()

            output = model(input_ids, attention_mask, token_type_ids)[0].cpu().numpy()

            y_pred.extend(output)

    predictions += [y_pred]

predictions = np.mean(predictions, 0).argmax(1)
sub = pd.read_csv('./Utils/data/sample.csv', dtype=object)  # 提交
sub['label'] = predictions
sub['label'] = sub['label'].apply(lambda x: ['A', 'B', 'C', 'D'][x])
sub.to_csv(args.input, index=False)
# np.mean(cv)
