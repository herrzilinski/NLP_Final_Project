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
MODEL_DIR = os.getcwd() + os.path.sep + 'Models' + os.path.sep

os.chdir(OR_PATH)  # Come back to the folder where the code resides , all files will be left on this directory

# %%
# df_train_raw = pd.read_csv(DATA_DIR+'train.csv')
df_raw = pd.read_csv(DATA_DIR + 'train.csv')
label_names = df_raw.columns[2:]
# df = df_raw.copy()
# df['labels'] = df_raw[label_names].values.tolist()
# df2 = df[['comment_text', 'labels']]
OUTPUTS_a = len(label_names)

# %%
####  Not gonna need anything above this it runs smoothly ####

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 200
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
NICKNAME = 'Electra'


# %%
class CustomDataset(Dataset):

    def __init__(self, text, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.comment_text = text
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = " ".join(self.comment_text.split())

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
            # 'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


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
        return self.act(pool)



# %%
input1 = ":Dear god this site is horrible."  # 0,0,0,0,0,0
input2 = ":If you have a look back at the source, the information I updated was the correct form. " \
         "I can only guess the source hadn't updated. " \
         "I shall update the information once again but thank you for your message."  # 0,0,0,0,0,0
input3 = "DJ Robinson is gay as hell! he sucks his dick so much!!!!!"  # 1,0,1,0,1,1
input4 = "::No, he is an arrogant, self serving, immature idiot. Get it right."  # 1,0,1,0,1,0
input5 = "== All admins must DIE! == "  # 1,0,0,1,0,0
input6 = ", you stupid retard, or I will kill you!!!   ]]"  # 1,1,1,1,1,0
input7 = "a wild ass contains dick in the pussy"  # 1,1,1,0,0,0
input8 = "There is a special place reserved for you in hell, you nazi piece of shit"  # Original, Toxic, insult
input9 = "Sorry, I shouldn't argue with you. I'm outta here... to sleep with your MAMA!"  # Original, Obscene Sarcasm


# testing_set = CustomDataset(df3, tokenizer, 200)
# testing_loader = DataLoader(testing_set, batch_size=1)
# %%
def toxicity_detection(text=None):
    if text is None:
        text = input('Please provide a message to classify:')
    if not isinstance(text, str):
        raise ValueError('message must be a string!')

    model = ElectraClass()
    model.load_state_dict(torch.load(MODEL_DIR + 'model_Electra_1.pt'), strict=False)
    model.eval()

    testing_set = CustomDataset(text, tokenizer, 200)
    testing_loader = DataLoader(testing_set, batch_size=1)

    for samp in testing_loader:
        ids = samp['ids']  # .to(device, dtype=torch.long)
        mask = samp['mask']  # .to(device, dtype=torch.long)
        token_type_ids = samp['token_type_ids']  # .to(device, dtype=torch.long)

    outputs = model(ids, mask, token_type_ids)
    predicted = outputs.detach().cpu().numpy()
    predicted[predicted >= 0.5] = 1
    predicted[predicted < 0.5] = 0
    res = [label_names[x] for x in range(6) if predicted.flatten()[x] == 1]

    if len(res) == 0:
        print('No toxicity detected.')
    else:
        print('Labels :', list(label_names.values))
        print('Output :', predicted)
        print('Prediction :', res)

    # outputs2 = model2(ids, mask, token_type_ids)
    # predicted2 = outputs2.detach().cpu().numpy()
    # predicted2[predicted2 >= 0.5] = 1
    # predicted2[predicted2 < 0.5] = 0


# %%
toxicity_detection(input9)
