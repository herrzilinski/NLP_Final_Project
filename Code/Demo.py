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
from sklearn.model_selection import train_test_split

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
# df_train_raw = pd.read_csv(DATA_DIR+'train.csv')
df_raw = pd.read_csv(DATA_DIR + 'train.csv')
label_names = df_raw.columns[2:]
df = df_raw.copy()
df['labels'] = df_raw[label_names].values.tolist()
df2 = df[['comment_text', 'labels']]
OUTPUTS_a = len(label_names)

# %%


#####  Not gonna need anything above this it runs smoothly ####


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 200
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
NICKNAME = 'Electra'



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


model1 = ElectraClass()
model2 = ElectraClass()

#model1.to(device)
model1.load_state_dict(torch.load('/home/ubuntu/NLP/Final Project/NLP_Final_Project/Code/model1.pt'))
model1.eval()
model2.eval()

# This is just to test, we will put the input in this dataframe format and muse fix labels ex: [0,0,0,0,0]
df3 = df2.iloc[6:7,:]
df3.reset_index(inplace=True, drop=True)

testing_set = CustomDataset(df3, tokenizer, 200)
testing_loader = DataLoader(testing_set, batch_size=1)

for samp in testing_loader:
    break

ids = samp['ids']                           #.to(device, dtype=torch.long)
mask = samp['mask']                         #.to(device, dtype=torch.long)
token_type_ids = samp['token_type_ids']     #.to(device, dtype=torch.long)

outputs1 = model1(ids, mask, token_type_ids)
outputs2 = model2(ids, mask, token_type_ids)









# %%
# num_testing_steps = EPOCHS * len(testing_loader)
# progress_bar = tqdm(range(num_testing_steps))
#
#
# def validation(epoch):
#     model.eval()
#     fin_targets = []
#     fin_outputs = []
#     with torch.no_grad():
#         for _, data in enumerate(testing_loader, 0):
#             ids = data['ids'].to(device, dtype=torch.long)
#             mask = data['mask'].to(device, dtype=torch.long)
#             token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
#             targets = data['targets'].to(device, dtype=torch.float)
#             outputs = model(ids, mask, token_type_ids)
#             fin_targets.extend(targets.cpu().detach().numpy().tolist())
#             fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
#             if _ % 5000 == 0:
#                 print(f'Epoch: {epoch}')
#                 progress_bar.update(1)
#
#     return fin_outputs, fin_targets
#
#
# for epoch in range(EPOCHS):
#     outputs, targets = validation(epoch)
#     outputs = np.array(outputs) >= THRESHOLD
#     accuracy = metrics.accuracy_score(targets, outputs)
#     f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
#     f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
#     print(f"Accuracy Score = {accuracy}")
#     print(f"F1 Score (Micro) = {f1_score_micro}")
#     print(f"F1 Score (Macro) = {f1_score_macro}")
#     targetss = np.array(targets)
#     fpr_micro, tpr_micro, _ = metrics.roc_curve(targetss.ravel(), outputs.ravel())
#     auc_micro = metrics.auc(fpr_micro, tpr_micro)
#     print(f"AUC Score (Micro) = {auc_micro}")

