import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from datasets import load_metric
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss

# References
# 1) https://towardsdatascience.com/transformers-for-multilabel-classification-71a1a0daf5e1
# 2) https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# 3)
'''
df = pd.read_csv('/home/ubuntu/NLP/Final Project/NLP_Final_Project/Data/train.csv')
label_cols = df.columns[2:]
df['labels'] = df[label_cols].values.tolist()

xtr, xte, ytr, yte = train_test_split(df['comment_text'], df['labels'])
df_tr = pd.DataFrame({'text':xtr, 'labels':ytr})
df_te = pd.DataFrame({'text':xte, 'labels':yte})
'''

class DatasetLoad(Dataset):
    def __init__(self):
        self.df = pd.read_csv('/home/ubuntu/NLP/Final Project/NLP_Final_Project/Data/train.csv')
        self.label_cols = self.df.columns[2:]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df['comment_text'][idx]
        labels = self.df[self.label_cols].values.tolist()[idx]
        sample= (text, labels)
        #if self.transform:
         #   sample = self.transform(sample)
        return sample




checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


'''
def tokenize_function(xdat, ydat):
    lst = xdat.values.tolist()
    tok = tokenizer(lst, truncation=True, padding=True, return_tensors='pt')
    tok['labels'] = torch.Tensor(ydat.values.tolist())
    return tok'''

def collate(batch):
    toks = {}
    inputid, type_id, attention, labels = [], [], [], []
    for txt, lbl in batch:
        tok = tokenizer(txt, truncation=True, padding='max_length')
        inputid.append(tok['input_ids'])
        type_id.append(tok['token_type_ids'])
        attention.append(tok['attention_mask'])
        labels.append(lbl)
        toks = {'input_ids': torch.tensor(inputid, dtype=torch.long),
                'token_type_ids': torch.tensor(type_id, dtype=torch.long),
                'attention_mask': torch.tensor(attention, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long)}
    return toks

a = DatasetLoad()


train_dataloader = DataLoader(a, shuffle=True, batch_size=8, collate_fn=collate)

eval_dataloader = DataLoader(a, batch_size=8, collate_fn=collate)

for batch in train_dataloader:
    break


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=6)
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

optimizer = AdamW(model.parameters(), lr=5e-5)
loss_func = BCEWithLogitsLoss()
#optimizer = AdamW(model.parameters(), lr=5e-5)

'''
# Forward pass for multilabel classification
outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
logits = outputs[0]
loss_func = BCEWithLogitsLoss()
loss = loss_func(logits.view(-1,num_labels),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation
# loss_func = BCELoss()
# loss = loss_func(torch.sigmoid(logits.view(-1,num_labels)),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation
train_loss_set.append(loss.item())
'''



num_epochs = 3

num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                              num_warmup_steps=0, num_training_steps=num_training_steps)
print(num_training_steps)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
device

progress_bar = tqdm(range(num_training_steps))

train_loss_set = []
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop('labels')
        outputs = model(**batch)
        logits = outputs[0]
        loss = loss_func(logits, labels.type_as(logits))
        train_loss_set.append(loss.item())
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

'''
metric = load_metric("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
'''