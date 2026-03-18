"""
Parallel corpus dataset for DE→EN machine translation.
Handles pre-tokenized data, vocabulary building, and batching.
"""

import os
import torch
from typing import List, Tuple, Dict, Optional
from collections import Counter
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# Special tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIALS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


class Vocabulary:
    """Word-level vocabulary with frequency-based filtering."""

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: List[str] = []
        self.word_count: Counter = Counter()

    def build(self, sentences: List[List[str]]):
        """Build vocabulary from list of tokenized sentences."""
        for sent in sentences:
            self.word_count.update(sent)

        self.idx2word = list(SPECIALS)
        self.word2idx = {tok: i for i, tok in enumerate(SPECIALS)}

        for word, count in self.word_count.most_common():
            if count >= self.min_freq:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices."""
        return [self.word2idx.get(t, UNK_IDX) for t in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices to tokens."""
        return [self.idx2word[i] if i < len(self.idx2word) else UNK_TOKEN for i in indices]

    def __len__(self):
        return len(self.idx2word)


class TranslationDataset(Dataset):
    """
    Dataset for parallel translation corpus.
    Expects pre-tokenized files where words are separated by spaces.
    """

    def __init__(
        self,
        src_file: str,
        trg_file: Optional[str],
        src_vocab: Optional[Vocabulary] = None,
        trg_vocab: Optional[Vocabulary] = None,
        min_freq: int = 2,
        max_length: int = 100,
    ):
        """
        Args:
            src_file: path to source language file (.de)
            trg_file: path to target language file (.en), None for test set
            src_vocab: pre-built source vocabulary (None = build from data)
            trg_vocab: pre-built target vocabulary (None = build from data)
            min_freq: minimum word frequency for vocabulary
            max_length: maximum sentence length in tokens (longer are truncated)
        """
        self.max_length = max_length

        # Read and tokenize source sentences
        with open(src_file, "r", encoding="utf-8") as f:
            self.src_sentences = [line.strip().split() for line in f if line.strip()]

        # Read and tokenize target sentences (if available)
        self.trg_sentences = None
        if trg_file is not None:
            with open(trg_file, "r", encoding="utf-8") as f:
                self.trg_sentences = [line.strip().split() for line in f if line.strip()]
            assert len(self.src_sentences) == len(self.trg_sentences), \
                f"Source ({len(self.src_sentences)}) and target ({len(self.trg_sentences)}) sizes mismatch"

        # Build or use provided vocabularies
        if src_vocab is None:
            self.src_vocab = Vocabulary(min_freq)
            self.src_vocab.build(self.src_sentences)
        else:
            self.src_vocab = src_vocab

        if trg_vocab is None and self.trg_sentences is not None:
            self.trg_vocab = Vocabulary(min_freq)
            self.trg_vocab.build(self.trg_sentences)
        else:
            self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx: int):
        src_tokens = self.src_sentences[idx][:self.max_length]
        src_indices = [SOS_IDX] + self.src_vocab.encode(src_tokens) + [EOS_IDX]
        src_tensor = torch.tensor(src_indices, dtype=torch.long)

        if self.trg_sentences is not None:
            trg_tokens = self.trg_sentences[idx][:self.max_length]
            trg_indices = [SOS_IDX] + self.trg_vocab.encode(trg_tokens) + [EOS_IDX]
            trg_tensor = torch.tensor(trg_indices, dtype=torch.long)
            return src_tensor, trg_tensor
        else:
            return src_tensor


def collate_fn(batch):
    """
    Collate function for DataLoader.
    Pads source and target sequences to the max length in the batch.
    """
    if isinstance(batch[0], tuple):
        src_batch, trg_batch = zip(*batch)
        src_lens = torch.tensor([len(s) for s in src_batch])
        trg_lens = torch.tensor([len(t) for t in trg_batch])
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
        trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=PAD_IDX)
        return src_padded, trg_padded, src_lens, trg_lens
    else:
        src_batch = batch
        src_lens = torch.tensor([len(s) for s in src_batch])
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
        return src_padded, src_lens


def load_data(data_dir: str, min_freq: int = 2, max_length: int = 100):
    """
    Load train, validation, and test datasets.
    Expected files in data_dir:
        train.de, train.en, val.de, val.en, test1.de
    """
    train_dataset = TranslationDataset(
        src_file=os.path.join(data_dir, "train.de"),
        trg_file=os.path.join(data_dir, "train.en"),
        min_freq=min_freq,
        max_length=max_length,
    )

    val_dataset = TranslationDataset(
        src_file=os.path.join(data_dir, "val.de"),
        trg_file=os.path.join(data_dir, "val.en"),
        src_vocab=train_dataset.src_vocab,
        trg_vocab=train_dataset.trg_vocab,
        max_length=max_length,
    )

    test_dataset = TranslationDataset(
        src_file=os.path.join(data_dir, "test1.de"),
        trg_file=None,
        src_vocab=train_dataset.src_vocab,
        max_length=max_length,
    )
    test_dataset.trg_vocab = train_dataset.trg_vocab

    print(f"Source vocab size: {len(train_dataset.src_vocab)}")
    print(f"Target vocab size: {len(train_dataset.trg_vocab)}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset
