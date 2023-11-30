import torch
from torch.utils.data import Dataset, DataLoader

# Assuming train_data is a list of sentences where each sentence is a list of tuples (word, label)
# Example: [('This', 'O'), ('is', 'O'), ('a', 'O'), ('sentence', 'B')]

class CustomDataset(Dataset):
    def __init__(self, data, vocab, label2index, max_length):
        self.data = data
        self.vocab = vocab
        self.label2index = label2index
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]

        # Extract words and labels
        words = [w for w, _ in sentence]
        labels = [l for _, l in sentence]

        # Convert words to embeddings
        word_embeddings = [self.vocab[w] for w in words]

        # Convert labels to indices
        label_indices = [self.label2index(l) for l in labels]
        temp_length = len(word_embeddings)
        # Pad or truncate to max_length
        if len(word_embeddings) < self.max_length:
            pad_length = self.max_length - len(word_embeddings)
            word_embeddings = word_embeddings + [self.vocab.get_default_index()] * pad_length
            label_indices = label_indices + [-1] * pad_length  # Assuming -1 is the index for padding
        else:
            word_embeddings = word_embeddings[:self.max_length]
            label_indices = label_indices[:self.max_length]

        # Create a mask
        mask = [1] * min(temp_length,self.max_length) + [0] * max(0,(self.max_length - temp_length))

        return {
            'word_embeddings': torch.tensor(word_embeddings),
            'label_indices': torch.tensor(label_indices),
            'mask': torch.tensor(mask),
            'max_length':temp_length
        }