import torch
import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        n_layers,
        dropout,
        future_input_size,
        output_size=1
    ):

        super().__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.future_embedding = nn.Sequential(
            nn.Linear(future_input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size + 32, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, history, future):

        h0 = torch.zeros(
            self.n_layers,
            history.size(0),
            self.hidden_size,
            device=history.device
        )

        c0 = torch.zeros(
            self.n_layers,
            history.size(0),
            self.hidden_size,
            device=history.device
        )

        _, (hidden, _) = self.lstm(history, (h0, c0))

        history_embedding = hidden[-1]

        future_embedding = self.future_embedding(future)

        combined = torch.cat(
            [history_embedding, future_embedding],
            dim=1
        )

        prediction = self.fc(combined)

        return prediction