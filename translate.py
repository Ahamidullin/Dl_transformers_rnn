"""
Translate the test set and write output file.
Also can be used to evaluate BLEU on validation set.
"""

import torch
from torch.utils.data import DataLoader

from dataset import load_data, collate_fn, PAD_IDX
from model import build_model
from train import translate_dataset, compute_bleu


def translate_test(checkpoint_path: str, data_dir: str, output_file: str,
                   embed_dim: int = 256, hidden_dim: int = 512,
                   attention_dim: int = 256, num_layers: int = 2,
                   dropout: float = 0.0,  # no dropout at inference
                   batch_size: int = 128, max_length: int = 100,
                   min_freq: int = 2, device_name: str = "cpu"):
    """Load checkpoint and translate test set."""
    device = torch.device(device_name)

    # Load data (need vocab from training data)
    train_dataset, val_dataset, test_dataset = load_data(data_dir, min_freq, max_length)

    # Build model
    model = build_model(
        src_vocab_size=len(train_dataset.src_vocab),
        trg_vocab_size=len(train_dataset.trg_vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        attention_dim=attention_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with BLEU {checkpoint['bleu']:.2f}")

    # Translate test set
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, collate_fn=collate_fn)
    translations = translate_dataset(model, test_loader, train_dataset.trg_vocab, device, max_length)

    # Write translations
    with open(output_file, "w", encoding="utf-8") as f:
        for line in translations:
            f.write(line + "\n")
    print(f"Wrote {len(translations)} translations to {output_file}")

    # Also evaluate on validation set
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)
    val_translations = translate_dataset(model, val_loader, train_dataset.trg_vocab, device, max_length)

    # Load references
    val_references = [" ".join(s) for s in val_dataset.trg_sentences]
    bleu = compute_bleu(val_translations, val_references)
    print(f"Validation BLEU: {bleu:.2f}")

    return translations


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="test1.de-en.en")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--attention_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    translate_test(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_file=args.output,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        max_length=args.max_length,
        min_freq=args.min_freq,
        device_name=args.device,
    )
