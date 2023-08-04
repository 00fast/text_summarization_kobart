import glob
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        context_info = data.get('dataset', {}).get('context_info')
        if context_info:
            return [(item.get('context'), item.get('summary')) for item in context_info if item.get('context') and item.get('summary')]
    return []

class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        return item

    def __len__(self):
        return len(self.labels)
