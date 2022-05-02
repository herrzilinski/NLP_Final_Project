import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, BertModel, BertConfig, ElectraTokenizer, ElectraModel
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BertTokenizer, BertModel, BertConfig
from tqdm.auto import tqdm
import os



# %%
OR_PATH = os.getcwd()
os.chdir("..")  # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
MODEL_DIR = os.getcwd() + os.path.sep + 'Models' + os.path.sep

os.chdir(OR_PATH)  # Come back to the folder where the code resides , all files will be left on this directory

# %%
OUTPUTS_a = 6
BATCH_SIZE = 16
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
MAX_LEN = 200
THRESHOLD = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
df_eval_labels = pd.read_csv(DATA_DIR + 'test_labels.csv')
df_eval_raw = pd.read_csv(DATA_DIR + 'test.csv')
label_names = df_eval_labels.columns[1:]

df_eval = pd.concat([df_eval_raw, df_eval_labels[label_names]], axis=1)
df_eval = df_eval[df_eval['toxic'] >= 0]
df_eval.reset_index(inplace=True)
df_eval['labels'] = df_eval[label_names].values.tolist()

# %%
class ElectraClass(torch.nn.Module):
    def __init__(self):
        super(ElectraClass, self).__init__()
        self.l1 = ElectraModel.from_pretrained('google/electra-small-discriminator', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(256, OUTPUTS_a)
        self.act = torch.nn.Sigmoid()

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1[0])
        output = self.l3(output_2)
        pool = output.mean(1)
        return pool


# %%
class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe['comment_text']
        self.targets = self.data['labels']
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            # max_length=self.max_len,
            truncation=True,
            padding='max_length',
            # pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }



# %%
def metrics_func(metrics, aggregates, y_true, y_pred):
    '''
    multiple functiosn of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    '''

    def f1_score_metric(y_true, y_pred, type):
        '''
            type = micro,macro,weighted,samples
        :param y_true:
        :param y_pred:
        :param average:
        :return: res
        '''
        res = f1_score(y_true, y_pred, average=type)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        return res

    def hamming_metric(y_true, y_pred):
        res = hamming_loss(y_true, y_pred)
        return res

    xcont = 1
    xsum = 0
    xavg = 0
    res_dict = {}
    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average =
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
             # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet = accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet = matthews_metric(y_true, y_pred)
        elif xm == 'hlm':
            xmet = -hamming_metric(y_true, y_pred)
        else:
            xmet = 0

        res_dict[xm] = xmet

        xsum = xsum + xmet
        xcont = xcont + 1

    if 'sum' in aggregates:
        res_dict['sum'] = xsum
    if 'avg' in aggregates and xcont > 0:
        res_dict['avg'] = xsum/xcont
    # Ask for arguments for each metric

    return res_dict


# %%
eval_params = {'batch_size': BATCH_SIZE,
               'shuffle': True,
               'num_workers': 4
               }
eval_set = CustomDataset(df_eval, tokenizer, MAX_LEN)
eval_loader = DataLoader(eval_set, **eval_params)



# %%
def eval(list_of_metrics, list_of_agg):

    model = ElectraClass()
    checkpoint = torch.load(MODEL_DIR + 'modelElectraSampled2.pt')
    model.load_state_dict(checkpoint, strict=False)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()

    pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

    test_loss, steps_test = 0, 0
    met_test = 0

    with torch.no_grad():

        with tqdm(total=len(eval_loader), desc="Evaluation") as pbar:

            for _, data in enumerate(eval_loader, 0):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                outputs = model(ids, mask, token_type_ids)

                loss = criterion(outputs, targets)
                test_loss += loss.item()

                steps_test += 1

                pbar.update(1)
                pbar.set_postfix_str(f"Eval Loss: {(test_loss / steps_test):.5f}")

                pred_logits = np.vstack((pred_logits, outputs.detach().cpu().numpy()))
                real_labels = np.vstack((real_labels, targets.detach().cpu().numpy()))

    pred_labels = pred_logits[1:]
    pred_labels[pred_labels >= THRESHOLD] = 1
    pred_labels[pred_labels < THRESHOLD] = 0

    test_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)

    avg_test_loss = test_loss / steps_test

    xstrres = ''
    for met, dat in test_metrics.items():
        xstrres = xstrres + ' Eval ' + met + ' {:.5f}'.format(dat)
    xstrres = xstrres + " - "
    print(xstrres)


if __name__ == '__main__':
    list_of_metrics = ['acc', 'hlm']
    list_of_agg = ['sum', 'avg']

    eval(list_of_metrics, list_of_agg)

