import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, BertModel, BertConfig, ElectraTokenizer, ElectraModel
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BertTokenizer, BertModel, BertConfig
from tqdm.auto import tqdm
import os
from torch import nn
from sklearn.model_selection import train_test_split
import sys
sys.path.append('/home/ubuntu/NLP/Final Project/NLP_Final_Project/Code')
from sampler import MultilabelBalancedRandomSampler


# Reference
# https://towardsdatascience.com/multi-label-emotion-classification-with-pytorch-huggingfaces-transformers-and-w-b-for-tracking-a060d817923

# %%
OR_PATH = os.getcwd()
os.chdir("..")  # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
# MODEL_DIR = os.getcwd() + os.path.sep + 'Models' + os.path.sep

os.chdir(OR_PATH)  # Come back to the folder where the code resides , all files will be left on this directory


# %%
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 200
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-05
THRESHOLD = 0.5
SAVE_MODEL = True
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
NICKNAME = 'Electra'

# %%
# df_train_raw = pd.read_csv(DATA_DIR+'train.csv')
df_raw = pd.read_csv(DATA_DIR + 'train.csv')
label_names = df_raw.columns[2:]
df = df_raw.copy()
df['labels'] = df_raw[label_names].values.tolist()
df2 = df[['comment_text', 'labels']]
OUTPUTS_a = len(label_names)

# Turning the labels into an array of size (sample_size x number_of_classes)
# Then it will be fed into MultilabelBalancedRandomSampler()
labels = df2['labels'].apply(lambda x: np.array(x)).values
labels2 = np.zeros((1, 6))
for i in range(len(labels)):
    labels2 = np.vstack((labels2,labels[i]))
labels_array = labels2[1:]

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
train_size = 0.75
train_dataset = df2.sample(frac=train_size, random_state=13)
test_dataset = df2.drop(train_dataset.index).reset_index(drop=True)
train_idx = train_dataset.index
test_idx = test_dataset.index

print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

# this will be used when DataLoader is called
train_sampler = MultilabelBalancedRandomSampler(labels_array, train_idx, class_choice="least_sampled")

training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

# %%
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'num_workers': 4
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
               'shuffle': True,
               'num_workers': 4
               }

training_loader = DataLoader(training_set, **train_params, sampler=train_sampler)
testing_loader = DataLoader(testing_set, **test_params)


# %%

class ElectraClass(torch.nn.Module):
    def __init__(self):
        super(ElectraClass, self).__init__()
        self.l1 = ElectraModel.from_pretrained('google/electra-small-discriminator', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(256, OUTPUTS_a)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1[0])
        output = self.l3(output_2)
        return output.mean(1)


model = ElectraClass()

if torch.cuda.device_count() > 1:
  model = nn.DataParallel(model)

model.to(device)

# %%
# def loss_fn(outputs, targets):
#     return torch.nn.BCEWithLogitsLoss()(outputs, targets)
criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
# num_training_steps = EPOCHS * len(training_loader)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

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
def train_val(train_ds, test_ds, list_of_metrics, list_of_agg, save_on, from_checkpoint=False):

    cont = 0
    train_loss_item = list([])
    test_loss_item = list([])

    pred_labels_per_hist = list([])

    model.phase = 0

    met_test_best = -1
    trigger_times = 0
    last_loss = 0

    for epoch in range(EPOCHS):
        train_loss, steps_train = 0, 0

        if from_checkpoint and epoch == 0:
            checkpoint = torch.load(f"model_{NICKNAME}.pt", map_location=device)
            model.load_state_dict(checkpoint)
            model.to(device)
        else:
            pass

        model.train()

        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        train_hist = list([])
        test_hist = list([])

        with tqdm(total=len(train_ds), desc="Epoch {}".format(epoch+1)) as pbar:

            for _, data in enumerate(train_ds, 0):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                optimizer.zero_grad()
                outputs = model(ids, mask, token_type_ids)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                cont += 1

                steps_train += 1

                train_loss_item.append([epoch, loss.item()])

                pred_labels_per = outputs.detach().to(torch.device('cpu')).numpy()

                if len(pred_labels_per_hist) == 0:
                    pred_labels_per_hist = pred_labels_per
                else:
                    pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])

                if len(train_hist) == 0:
                    train_hist = targets.cpu().numpy()
                else:
                    train_hist = np.vstack([train_hist, targets.cpu().numpy()])

                pbar.update(1)
                pbar.set_postfix_str(f"Train Loss: {(train_loss / steps_train):.5f}")

                pred_logits = np.vstack((pred_logits, outputs.detach().cpu().numpy()))
                real_labels = np.vstack((real_labels, targets.cpu().numpy()))

        pred_labels = pred_logits[1:]
        pred_labels[pred_labels >= THRESHOLD] = 1
        pred_labels[pred_labels < THRESHOLD] = 0

        # Metric Evaluation
        train_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)

        xstrres = "Epoch {}: ".format(epoch+1)
        for met, dat in train_metrics.items():
            xstrres = xstrres +' Train '+ met + ' {:.5f}'.format(dat)

        xstrres = xstrres + " - "
        print(xstrres)

        ## Finish with Training

        ## Testing the model

        model.eval()

        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        test_loss, steps_test = 0, 0
        met_test = 0

        with torch.no_grad():

            with tqdm(total=len(test_ds), desc="Epoch {}".format(epoch+1)) as pbar:

                for _, data in enumerate(test_ds, 0):
                    ids = data['ids'].to(device, dtype=torch.long)
                    mask = data['mask'].to(device, dtype=torch.long)
                    token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                    targets = data['targets'].to(device, dtype=torch.float)
                    outputs = model(ids, mask, token_type_ids)

                    loss = criterion(outputs, targets)
                    test_loss += loss.item()
                    cont += 1

                    steps_test += 1

                    test_loss_item.append([epoch, loss.item()])

                    pred_labels_per = outputs.detach().to(torch.device('cpu')).numpy()

                    if len(pred_labels_per_hist) == 0:
                        pred_labels_per_hist = pred_labels_per
                    else:
                        pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])

                    if len(test_hist) == 0:
                        test_hist = targets.cpu().numpy()
                    else:
                        test_hist = np.vstack([test_hist, targets.cpu().numpy()])

                    pbar.update(1)
                    pbar.set_postfix_str(f"Test Loss: {(test_loss / steps_test):.5f}")

                    pred_logits = np.vstack((pred_logits, outputs.detach().cpu().numpy()))
                    real_labels = np.vstack((real_labels, targets.cpu().numpy()))

        pred_labels = pred_logits[1:]
        pred_labels[pred_labels >= THRESHOLD] = 1
        pred_labels[pred_labels < THRESHOLD] = 0

        test_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)

        avg_test_loss = test_loss / steps_test

        xstrres = "Epoch {}: ".format(epoch+1)
        for met, dat in test_metrics.items():
            xstrres = xstrres + ' Test ' + met + ' {:.5f}'.format(dat)
            if met == save_on:
                met_test = dat

        xstrres = xstrres + " - "
        print(xstrres)

        if met_test > met_test_best and SAVE_MODEL:
            torch.save(model.state_dict(), "model_{}.pt".format(NICKNAME))
            print("The model has been saved!")
            met_test_best = met_test

        # early stopping
        if avg_test_loss > last_loss:
            # if avg_test_loss < 0.35:
            #     break
            trigger_times += 1
            print('Trigger Times:', trigger_times)
            if trigger_times >= 5:
                print('Early stopping!\nStart to test process.')
                break
        else:
            print('Trigger Times: 0')
            trigger_times = 0
        last_loss = avg_test_loss

        # learning rate scheduler
        scheduler.step(met_test_best)


# %%

if __name__ == '__main__':
    list_of_metrics = ['acc', 'hlm']
    list_of_agg = ['sum', 'avg']
    train_val(training_loader, testing_loader, list_of_metrics, list_of_agg, save_on='sum', from_checkpoint=False)

