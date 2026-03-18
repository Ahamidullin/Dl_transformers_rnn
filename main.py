"""
Main entry point for training and translating.
Usage:
    python main.py                     # train from scratch
    python main.py --translate_only    # translate test set using saved checkpoint
"""

import argparse
import torch
from torch.utils.data import DataLoader

from dataset import load_data, collate_fn
from model import build_model
from train import train, translate_dataset, compute_bleu


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="DE→EN Translation with GRU + Attention")
    # Data
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=100)
    # Model
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--attention_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    # Training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--tf_start", type=float, default=1.0,
                        help="Teacher forcing ratio at start")
    parser.add_argument("--tf_end", type=float, default=0.5,
                        help="Teacher forcing ratio at end")
    # Paths
    parser.add_argument("--save_path", type=str, default="best_model.pt")
    parser.add_argument("--output", type=str, default="test1.de-en.en")
    # Modes
    parser.add_argument("--translate_only", action="store_true",
                        help="Only translate test set, no training")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    device = torch.device(args.device) if args.device else get_device()
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    train_dataset, val_dataset, test_dataset = load_data(
        args.data_dir, args.min_freq, args.max_length
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    # Build model
    model = build_model(
        src_vocab_size=len(train_dataset.src_vocab),
        trg_vocab_size=len(train_dataset.trg_vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # Validation references
    val_references = [" ".join(s) for s in val_dataset.trg_sentences]

    if not args.translate_only:
        # Train
        print("\n=== Training ===")
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            trg_vocab=train_dataset.trg_vocab,
            val_references=val_references,
            num_epochs=args.num_epochs,
            lr=args.lr,
            clip=args.clip,
            teacher_forcing_start=args.tf_start,
            teacher_forcing_end=args.tf_end,
            device=device,
            save_path=args.save_path,
        )

    # Load best model and translate test set
    print("\n=== Translating test set ===")
    checkpoint = torch.load(args.save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best model from epoch {checkpoint['epoch']} (BLEU {checkpoint['bleu']:.2f})")

    # Translate test
    test_translations = translate_dataset(
        model, test_loader, train_dataset.trg_vocab, device, args.max_length
    )
    with open(args.output, "w", encoding="utf-8") as f:
        for line in test_translations:
            f.write(line + "\n")
    print(f"Wrote {len(test_translations)} test translations to {args.output}")

    # Final validation BLEU
    val_translations = translate_dataset(
        model, val_loader, train_dataset.trg_vocab, device, args.max_length
    )
    bleu = compute_bleu(val_translations, val_references)
    print(f"\nFinal Validation BLEU: {bleu:.2f}")

    # Print some examples
    print("\n=== Sample translations ===")
    for i in range(min(5, len(val_translations))):
        src = " ".join(val_dataset.src_sentences[i])
        print(f"SRC: {src}")
        print(f"HYP: {val_translations[i]}")
        print(f"REF: {val_references[i]}")
        print()


if __name__ == "__main__":
    main()
