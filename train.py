"""
Training loop for seq2seq translation model.
Includes teacher forcing, gradient clipping, BLEU evaluation.
"""

import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, List
import sacrebleu

from dataset import PAD_IDX, EOS_IDX, collate_fn
from model import Seq2Seq


def train_epoch(model: Seq2Seq, loader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, clip: float, teacher_forcing_ratio: float,
                device: torch.device) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    epoch_loss = 0.0

    for src, trg, src_lens, trg_lens in loader:
        src, trg = src.to(device), trg.to(device)
        src_lens = src_lens.to(device)

        optimizer.zero_grad()

        output = model(src, src_lens, trg, teacher_forcing_ratio)
        # output: (B, trg_len-1, vocab_size)
        # trg:    (B, trg_len) — skip first token (SOS)

        output = output.reshape(-1, output.size(-1))
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


@torch.no_grad()
def evaluate(model: Seq2Seq, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> float:
    """Evaluate model. Returns average loss."""
    model.eval()
    epoch_loss = 0.0

    for src, trg, src_lens, trg_lens in loader:
        src, trg = src.to(device), trg.to(device)
        src_lens = src_lens.to(device)

        output = model(src, src_lens, trg, teacher_forcing_ratio=0.0)

        output = output.reshape(-1, output.size(-1))
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def translate_dataset(model: Seq2Seq, loader: DataLoader, trg_vocab,
                      device: torch.device, max_length: int = 128) -> List[str]:
    """Translate all sentences in a dataloader and return string translations."""
    model.eval()
    all_translations = []

    for batch in loader:
        if isinstance(batch, tuple) and len(batch) == 4:
            src, _, src_lens, _ = batch
        else:
            src, src_lens = batch

        src = src.to(device)
        src_lens = src_lens.to(device)

        decoded = model.translate(src, src_lens, max_length=max_length)

        for token_ids in decoded:
            tokens = trg_vocab.decode(token_ids)
            all_translations.append(" ".join(tokens))

    return all_translations


def compute_bleu(hypotheses: List[str], references: List[str]) -> float:
    """Compute BLEU score using sacrebleu."""
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="none")
    return bleu.score


def train(model: Seq2Seq, train_loader: DataLoader, val_loader: DataLoader,
          trg_vocab, val_references: List[str],
          num_epochs: int = 25, lr: float = 1e-3, clip: float = 1.0,
          teacher_forcing_start: float = 1.0, teacher_forcing_end: float = 0.5,
          device: torch.device = torch.device("cpu"),
          save_path: str = "best_model.pt"):
    """
    Full training loop with BLEU evaluation and model saving.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_bleu = -1.0
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        # Anneal teacher forcing
        tf_ratio = teacher_forcing_start - (teacher_forcing_start - teacher_forcing_end) * (epoch - 1) / max(num_epochs - 1, 1)

        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, clip, tf_ratio, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start

        scheduler.step(val_loss)

        # Compute BLEU on validation set
        translations = translate_dataset(model, val_loader, trg_vocab, device)
        bleu_score = compute_bleu(translations, val_references)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:02d}/{num_epochs} | "
              f"TF: {tf_ratio:.2f} | LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"PPL: {math.exp(val_loss):.2f} | BLEU: {bleu_score:.2f} | "
              f"Time: {elapsed:.1f}s")

        # Print sample translations every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            print("  Sample translations:")
            for i in range(min(3, len(translations))):
                print(f"    HYP: {translations[i]}")
                print(f"    REF: {val_references[i]}")
                print()

        # Save best model by BLEU
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "bleu": best_bleu,
                "val_loss": val_loss,
            }, save_path)
            print(f"  ★ New best BLEU: {best_bleu:.2f} — model saved!")

    print(f"\nTraining finished. Best BLEU: {best_bleu:.2f}")
    return best_bleu
