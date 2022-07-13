from torch.utils.data import Dataset
import torch


class Text2TextDataset(Dataset):

    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_row = self.data.iloc[idx]
        text = data_row.text
        label = data_row.label

        encodings_dict = self.tokenizer(text + str(label), truncation=True, max_length=self.max_length,
                                        padding="max_length", return_tensors="pt")

        joint_attn = encodings_dict['attention_mask']
        text_only_attn = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                        padding="max_length", return_tensors="pt")['attention_mask']
        label_attn = torch.logical_xor(joint_attn, text_only_attn)
        labels = torch.where(label_attn, encodings_dict['input_ids'], -100)

        return dict(
            input_ids=encodings_dict['input_ids'].flatten(),
            attention_mask=joint_attn.flatten(),
            labels=labels.flatten()
        )
