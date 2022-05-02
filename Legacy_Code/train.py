import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
from transformers import BertTokenizer, BertModel, BertConfig
import torch
from tqdm.auto import tqdm
from datasets import load_metric
import os


OR_PATH = os.getcwd()
os.chdir("..")  # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
# MODEL_DIR = os.getcwd() + os.path.sep + 'Models' + os.path.sep

os.chdir(OR_PATH)  # Come back to the folder where the code resides , all files will be left on this directory


# %%
MAX_LEN = 200
BATCH_SIZE = 8
EPOCHS = 3
LR = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# %%
df_train_raw = pd.read_csv(DATA_DIR+'train.csv')
label_names = df_train_raw.columns[2:]
df_train = df_train_raw.copy()
df_train['labels'] = df_train_raw[label_names].values.tolist()
df_train = df_train[['comment_text', 'labels']]

# %%
class Custom_Data_loader(Dataset):
    def __init__(self, data, tokenizer, maxlen):
        self.df = data
        self.dftext = data['comment_text']
        self.targets = data['labels']
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        text = str(self.dftext[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            # max_length=self.maxlen,
            # truncation=True,
            pad_to_max_length=True,
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
train_dataset, test_dataset = train_test_split(df_train, test_size=0.2, shuffle=True, random_state=1)
train_dataset = train_dataset.reset_index()
test_dataset = test_dataset.reset_index()

print("FULL Dataset: {}".format(df_train.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = Custom_Data_loader(train_dataset, tokenizer, MAX_LEN)
testing_set = Custom_Data_loader(test_dataset, tokenizer, MAX_LEN)

# %%
training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
testing_loader = DataLoader(testing_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# %%
class BERTFineTune(torch.nn.Module):
    def __init__(self):
        super(BERTFineTune, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.l2 = torch.nn.Dropout(0.5)
        self.l3 = torch.nn.Linear(768, 6)

    def forward(self, ids, mask, token_type_ids):
        _, x1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        x2 = self.l2(x1)
        output = self.l3(x2)
        return output


model = BERTFineTune()
model.to(device)


# %%
# def loss_fn(outputs, targets):
#     return torch.nn.BCEWithLogitsLoss()(outputs, targets)
criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

# %%
def train(epoch):
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets']

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        if _ % 5000 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# def train(dataloader):
#     model.train()
#     total_acc, total_count = 0, 0
#     log_interval = 500
#     start_time = time.time()

    # for idx, data in enumerate(dataloader, 0):
    #     ids = data['ids'].to(device, dtype=torch.long)
    #     mask = data['mask'].to(device, dtype=torch.long)
    #     token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
    #     targets = data['targets'].to(device, dtype=torch.float)
    #
    #     optimizer.zero_grad()
    #     predicted_label = model(ids, mask, token_type_ids)
    #     loss = criterion(predicted_label, targets)
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    #     optimizer.step()
    #     total_acc += (predicted_label.argmax(1) == targets).sum().item()
    #     total_count += targets.size(0)
    #     if idx % log_interval == 0 and idx > 0:
    #         elapsed = time.time() - start_time
    #         print('| epoch {:3d} | {:5d}/{:5d} batches '
    #               '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
    #                                           total_acc / total_count))
    #         total_acc, total_count = 0, 0
    #         start_time = time.time()




for epoch in range(EPOCHS):
    train(epoch)


# %%
def validation(epoch):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


for epoch in range(EPOCHS):
    outputs, targets = validation(epoch)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")