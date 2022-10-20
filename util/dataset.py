import json
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertTokenizer


def make_data(file_path, args):
    dataset = LawDataset(file_path, args.bert_path)
    return DataLoader(dataset,
                      batch_size=args.batch_size,
                      shuffle=True,
                      num_workers=4,
                      collate_fn=my_collate)


def my_collate(batch):
    text = [item[0] for item in batch]
    label = [item[1] for item in batch]
    lenth = [item[2] for item in batch]

    label = torch.LongTensor(label)
    label = torch.nn.functional.one_hot(label, num_classes=6)

    return text, torch.LongTensor(lenth), label


class LawDataset(Dataset):
    def __init__(self, file, bert_path):
        super(LawDataset, self).__init__()
        self.data = []
        self.label = []
        self.tokenizer = BertTokenizer.from_pretrained(bert_path,
                                                       do_lower_case=False)
        with open(file, "r", encoding='utf-8') as f:
            all = json.load(f)
        for i in all:
            self.data.append(i['fact'])
            self.label.append(i['charge'])
        self.dic = {5: 0, 32: 1, 40: 2, 48: 3, 68: 4, 78: 5}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token = []
        a = []
        for i in self.data[idx]:
            if i == '。' or i == '，':
                token.append(a)
                a = []
            else:
                a.append(self.tokenizer.convert_tokens_to_ids(i))
        if len(a) != 0:
            token.append(a)

        label_idx = self.label[idx]

        lenth = [len(i) for i in token]
        max_lenth = max(lenth)
        token = [i + [0]*(max_lenth-len(i)) for i in token]

        return token, self.dic[label_idx], len(token)
