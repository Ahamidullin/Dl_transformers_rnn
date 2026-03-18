"""
Seq2Seq model with Bidirectional GRU Encoder, Bahdanau Attention, and GRU Decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    """Bidirectional GRU encoder."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Project bidirectional hidden to decoder-compatible size
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor):
        """
        Args:
            src: (batch, src_len) — source token indices
            src_lens: (batch,) — actual lengths
        Returns:
            outputs: (batch, src_len, hidden_dim * 2) — encoder outputs
            hidden:  (num_layers, batch, hidden_dim) — last hidden state for decoder
        """
        embedded = self.dropout(self.embedding(src))  # (B, S, E)

        packed = pack_padded_sequence(embedded, src_lens.cpu().clamp(min=1),
                                     batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)  # (B, S, H*2)

        # hidden: (num_layers * 2, B, H) → combine forward/backward
        # Reshape to (num_layers, 2, B, H), then cat directions
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)  # (num_layers, B, H*2)
        hidden = torch.tanh(self.fc_hidden(hidden))  # (num_layers, B, H)

        return outputs, hidden


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention mechanism."""

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        self.W_enc = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.W_dec = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.V = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor,
                mask: torch.Tensor):
        """
        Args:
            decoder_hidden: (batch, hidden_dim) — current decoder hidden state
            encoder_outputs: (batch, src_len, encoder_dim) — all encoder outputs
            mask: (batch, src_len) — True for PAD positions
        Returns:
            context: (batch, encoder_dim) — weighted sum of encoder outputs
            attn_weights: (batch, src_len) — attention weights
        """
        # (B, S, A) + (B, 1, A) → (B, S, A)
        energy = torch.tanh(
            self.W_enc(encoder_outputs) + self.W_dec(decoder_hidden).unsqueeze(1)
        )
        scores = self.V(energy).squeeze(2)  # (B, S)

        # Mask padding
        scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=1)  # (B, S)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, enc_dim)

        return context, attn_weights


class Decoder(nn.Module):
    """GRU decoder with Bahdanau attention."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 encoder_dim: int, attention_dim: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(encoder_dim, hidden_dim, attention_dim)

        # GRU input = embedding + context vector
        self.rnn = nn.GRU(
            embed_dim + encoder_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output projection: GRU output + context + embedding → vocab
        self.fc_out = nn.Linear(hidden_dim + encoder_dim + embed_dim, vocab_size)

    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor,
                encoder_outputs: torch.Tensor, mask: torch.Tensor):
        """
        One decoding step.
        Args:
            input_token: (batch,) — current input token indices
            hidden: (num_layers, batch, hidden_dim) — decoder hidden state
            encoder_outputs: (batch, src_len, encoder_dim) — encoder outputs
            mask: (batch, src_len) — True for PAD positions
        Returns:
            prediction: (batch, vocab_size) — logits for next token
            hidden: updated hidden state
            attn_weights: attention weights for this step
        """
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))  # (B, 1, E)

        # Use top layer hidden state for attention
        context, attn_weights = self.attention(
            hidden[-1], encoder_outputs, mask
        )  # context: (B, enc_dim)

        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)  # (B, 1, E+enc_dim)
        output, hidden = self.rnn(rnn_input, hidden)  # output: (B, 1, H)

        # Combine GRU output, context, and embedding for prediction
        output = output.squeeze(1)  # (B, H)
        prediction = self.fc_out(
            torch.cat([output, context, embedded.squeeze(1)], dim=1)
        )  # (B, vocab_size)

        return prediction, hidden, attn_weights


class Seq2Seq(nn.Module):
    """Seq2Seq model combining encoder, decoder, and attention."""

    def __init__(self, encoder: Encoder, decoder: Decoder,
                 pad_idx: int = 0, sos_idx: int = 1, eos_idx: int = 2):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def create_mask(self, src: torch.Tensor):
        """Create boolean mask: True where src == pad_idx."""
        return src == self.pad_idx

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor,
                trg: torch.Tensor, teacher_forcing_ratio: float = 0.5):
        """
        Args:
            src: (batch, src_len)
            src_lens: (batch,)
            trg: (batch, trg_len)
            teacher_forcing_ratio: probability of using ground truth as next input
        Returns:
            outputs: (batch, trg_len - 1, vocab_size) — predictions for tokens 1..T
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.vocab_size

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len - 1, trg_vocab_size, device=src.device)

        # Encode
        encoder_outputs, hidden = self.encoder(src, src_lens)
        mask = self.create_mask(src)

        # First input is <sos>
        input_token = trg[:, 0]  # (B,)

        for t in range(1, trg_len):
            prediction, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, mask)
            outputs[:, t - 1] = prediction

            # Teacher forcing
            if random.random() < teacher_forcing_ratio:
                input_token = trg[:, t]
            else:
                input_token = prediction.argmax(dim=1)

        return outputs

    @torch.inference_mode()
    def translate(self, src: torch.Tensor, src_lens: torch.Tensor,
                  max_length: int = 128):
        """
        Greedy decoding for inference.
        Args:
            src: (batch, src_len)
            src_lens: (batch,)
            max_length: maximum output length
        Returns:
            decoded_tokens: list of lists of token indices (without SOS, up to EOS)
        """
        self.eval()
        batch_size = src.size(0)

        encoder_outputs, hidden = self.encoder(src, src_lens)
        mask = self.create_mask(src)

        input_token = torch.full((batch_size,), self.sos_idx,
                                 dtype=torch.long, device=src.device)

        decoded = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for _ in range(max_length):
            prediction, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, mask)
            top1 = prediction.argmax(dim=1)  # (B,)

            for i in range(batch_size):
                if not finished[i]:
                    tok = top1[i].item()
                    if tok == self.eos_idx:
                        finished[i] = True
                    else:
                        decoded[i].append(tok)

            if all(finished):
                break

            input_token = top1

        return decoded

    @torch.inference_mode()
    def beam_search_translate(self, src: torch.Tensor, src_lens: torch.Tensor,
                              beam_size: int = 5, max_length: int = 128,
                              length_penalty: float = 0.6):
        """
        Beam search decoding for inference. Processes one sentence at a time.
        Args:
            src: (batch, src_len)
            src_lens: (batch,)
            beam_size: number of beams
            max_length: maximum output length
            length_penalty: alpha for length normalization (score / len^alpha)
        Returns:
            decoded_tokens: list of lists of token indices
        """
        self.eval()
        batch_size = src.size(0)
        all_decoded = []

        for i in range(batch_size):
            # Process one sentence at a time
            src_i = src[i:i+1]             # (1, src_len)
            src_len_i = src_lens[i:i+1]    # (1,)

            encoder_outputs, hidden = self.encoder(src_i, src_len_i)
            mask = self.create_mask(src_i)

            # Each beam: (log_prob, tokens, hidden_state)
            # Start with <sos>
            input_token = torch.full((1,), self.sos_idx, dtype=torch.long, device=src.device)
            prediction, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, mask)
            log_probs = torch.log_softmax(prediction[0], dim=0)  # (vocab,)

            # Initialize beams with top-k first tokens
            topk_probs, topk_idx = log_probs.topk(beam_size)
            beams = []
            for k in range(beam_size):
                tok = topk_idx[k].item()
                if tok == self.eos_idx:
                    beams.append((topk_probs[k].item(), [], hidden, True))
                else:
                    beams.append((topk_probs[k].item(), [tok], hidden, False))

            # Expand beams
            for _ in range(max_length - 1):
                all_finished = all(b[3] for b in beams)
                if all_finished:
                    break

                candidates = []
                for log_prob, tokens, h, finished in beams:
                    if finished:
                        candidates.append((log_prob, tokens, h, True))
                        continue

                    input_token = torch.tensor([tokens[-1]], dtype=torch.long, device=src.device)
                    prediction, new_h, _ = self.decoder(input_token, h, encoder_outputs, mask)
                    step_log_probs = torch.log_softmax(prediction[0], dim=0)

                    topk_probs, topk_idx = step_log_probs.topk(beam_size)
                    for k in range(beam_size):
                        tok = topk_idx[k].item()
                        new_log_prob = log_prob + topk_probs[k].item()
                        if tok == self.eos_idx:
                            candidates.append((new_log_prob, tokens, new_h, True))
                        else:
                            candidates.append((new_log_prob, tokens + [tok], new_h, False))

                # Score with length penalty and keep top beam_size
                def score(beam):
                    lp = max(len(beam[1]), 1) ** length_penalty
                    return beam[0] / lp

                candidates.sort(key=score, reverse=True)
                beams = candidates[:beam_size]

            # Best beam
            beams.sort(key=score, reverse=True)
            all_decoded.append(beams[0][1])

        return all_decoded


def build_model(src_vocab_size: int, trg_vocab_size: int,
                embed_dim: int = 256, hidden_dim: int = 512,
                attention_dim: int = 256, num_layers: int = 2,
                dropout: float = 0.3):
    """Create a Seq2Seq model with the given hyperparameters."""
    encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
    decoder = Decoder(
        trg_vocab_size, embed_dim, hidden_dim,
        encoder_dim=hidden_dim * 2,  # bidirectional
        attention_dim=attention_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    model = Seq2Seq(encoder, decoder)

    # Initialize weights
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.zeros_(param)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params:,} trainable parameters")

    return model
