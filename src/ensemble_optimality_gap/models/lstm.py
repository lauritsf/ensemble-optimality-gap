import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Attention(nn.Module):
    """A self-attention module for sequence classification."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_layer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs):
        # lstm_outputs: (batch_size, seq_len, hidden_dim)
        energy = torch.tanh(self.attention_layer(lstm_outputs)).squeeze(2)
        attention_weights = torch.softmax(energy, dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_outputs).squeeze(1)
        return context_vector


class BiLSTM(nn.Module):
    """
    A BiLSTM model for sequence classification with an optional attention mechanism.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        pad_idx: int,
        num_layers: int = 1,
        bidirectional: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Initialize attention only if it's going to be used
        if self.use_attention:
            self.attention = Attention(lstm_output_dim)

        self.output_layer = nn.Linear(lstm_output_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        # Get sequence lengths from the attention mask
        sequence_lengths = attention_mask.sum(dim=1).cpu()

        # Embedding -> Packing
        embedded = self.embedding(input_ids)
        packed_embedded = pack_padded_sequence(embedded, sequence_lengths, batch_first=True, enforce_sorted=False)

        # LSTM pass captures both the full output sequence and the final hidden states
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # --- Conditional Logic for Attention ---
        if self.use_attention:
            # Unpack the sequence to get hidden states for all time steps
            lstm_outputs, _ = pad_packed_sequence(packed_output, batch_first=True)
            # Pass through attention to get the context vector
            final_representation = self.attention(lstm_outputs)
        else:
            # Use the final hidden state (standard BiLSTM behavior)
            if self.lstm.bidirectional:
                # Concatenate the final forward and backward hidden states
                final_representation = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            else:
                # Just use the final hidden state of the last layer
                final_representation = hidden[-1, :, :]

        # Pass the chosen representation through the final output layer
        return self.output_layer(final_representation)
