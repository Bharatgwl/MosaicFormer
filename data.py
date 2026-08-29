"""Datasets and synthetic recall data for MosaicFormer experiments.

Copyright (c) 2026 Bharat Gusaiwal. All rights reserved.
"""

import torch
from torch.utils.data import DataLoader, Dataset


class GPTDatasetV1(Dataset):
    """Sliding-window next-token-prediction dataset from the GPT notebooks."""
    def __init__(self, token_ids, context_length, stride):
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(token_ids) - context_length, stride):
            input_chunk = token_ids[i:i + context_length]
            target_chunk = token_ids[i + 1:i + context_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


class LazyGPTDataset(Dataset):
    """Next-token dataset that slices token_ids on demand.

    This avoids materializing every input/target window as a separate tensor,
    which matters for larger text files such as WikiText-103.
    """
    def __init__(self, token_ids, context_length, stride):
        self.token_ids = token_ids
        self.context_length = context_length
        self.stride = stride
        self.num_examples = max(0, (len(token_ids) - context_length + stride - 1) // stride)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        start = index * self.stride
        end = start + self.context_length
        input_chunk = self.token_ids[start:end]
        target_chunk = self.token_ids[start + 1:end + 1]
        return torch.tensor(input_chunk, dtype=torch.long), torch.tensor(target_chunk, dtype=torch.long)


def create_dataloader(token_ids, context_length, stride, batch_size, shuffle, drop_last, lazy=True):
    dataset_class = LazyGPTDataset if lazy else GPTDatasetV1
    dataset = dataset_class(token_ids, context_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
    )


def split_token_ids(token_ids, train_ratio=0.8):
    split_index = int(len(token_ids) * train_ratio)
    return token_ids[:split_index], token_ids[split_index:]


def make_needle_recall_data(num_examples, context_length, vocab_size, seed=123):
    """Creates a controlled long-context task.

    The final token is a copy of a value placed near the beginning. After
    training on this task, final-token accuracy shows how well long history is
    retained. Reserve IDs 0-3 for structure and sample content from 4 onward.
    """
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(4, vocab_size, (num_examples, context_length), generator=generator)
    keys = torch.randint(4, vocab_size, (num_examples,), generator=generator)
    values = torch.randint(4, vocab_size, (num_examples,), generator=generator)

    input_ids[:, 0] = 1
    input_ids[:, 1] = keys
    input_ids[:, 2] = values
    input_ids[:, -3] = 2
    input_ids[:, -2] = keys
    input_ids[:, -1] = 3
    targets = values
    return input_ids, targets
