# models/st_gcn.py
import torch
import torch.nn as nn
import numpy as np

def get_adjacency_matrix():
    A = np.zeros((4, 4), dtype=np.float32)
    edges = [
        (0, 0), (1, 1), (2, 2), (3, 3), 
        (0, 1), (1, 0),                 
        (0, 2), (2, 0),                 
        (1, 2), (2, 1),                 
        (0, 3), (3, 0),                 
        (1, 3), (3, 1)                  
    ]
    
    for i, j in edges:
        A[i, j] = 1.0
        
    D = np.diag(np.sum(A, axis=1) ** (-0.5))
    A_norm = D @ A @ D
    
    return torch.tensor(A_norm, dtype=torch.float32)

class STGCN_Block(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        stride=1
    ):
        super().__init__()

        self.A = nn.Parameter(
            A.clone(),
            requires_grad=True
        )

        self.gcn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

        self.spatial_bn = nn.BatchNorm2d(
            out_channels
        )

        self.tcn = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(9, 1),
            padding=(4, 0),
            stride=(stride, 1),
        )

        self.temporal_bn = nn.BatchNorm2d(
            out_channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        if (
            in_channels != out_channels
            or stride != 1
        ):
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(
                    out_channels
                ),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):

        res = self.residual(x)

        x = self.gcn(x)

        x = torch.matmul(
            x,
            self.A
        )

        x = self.relu(
            self.spatial_bn(x)
        )

        x = self.temporal_bn(
            self.tcn(x)
        )

        return self.relu(
            x + res
        )


class TKD_STGCN(nn.Module):

    def __init__(
        self,
        num_classes,
        A
    ):
        super().__init__()

        self.data_bn = nn.BatchNorm2d(2)

        self.layer1 = STGCN_Block(
            2,
            32,
            A
        )

        self.layer2 = STGCN_Block(
            32,
            64,
            A
        )

        self.drop = nn.Dropout(0.5)

        self.fc = nn.Linear(
            64,
            num_classes
        )

    def forward(
        self,
        x,
        lengths
    ):

        x = self.data_bn(x)

        x = self.layer1(x)

        x = self.layer2(x)

        B, C, T, V = x.shape

        mask = (
            torch.arange(
                T,
                device=x.device
            )[None, :]
            <
            lengths.to(x.device)[:, None]
        )

        mask = mask.view(
            B,
            1,
            T,
            1
        ).float()

        x = x * mask

        x = x.sum(
            dim=(2, 3)
        )

        valid_count = (
            lengths.to(x.device).float()
            * V
        ).unsqueeze(1)

        x = x / valid_count.clamp_min(1.0)

        x = self.drop(x)

        return self.fc(x)


class TKD_LSTM(nn.Module):

    def __init__(
        self,
        num_classes
    ):
        super().__init__()

        self.data_norm = nn.LayerNorm(8)

        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
        )

        self.drop = nn.Dropout(0.5)

        self.fc = nn.Linear(
            64,
            num_classes
        )

    def forward(
        self,
        x,
        lengths
    ):

        B, C, T, V = x.shape

        x = (
            x.permute(
                0,
                2,
                3,
                1
            )
            .contiguous()
            .view(
                B,
                T,
                V * C
            )
        )

        x = self.data_norm(x)

        packed_x = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (h_n, _) = self.lstm(
            packed_x
        )

        out = self.drop(
            h_n[-1]
        )

        return self.fc(out)


class TKD_GRU(nn.Module):

    def __init__(
        self,
        num_classes
    ):
        super().__init__()

        self.data_norm = nn.LayerNorm(8)

        self.gru = nn.GRU(
            input_size=8,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
        )

        self.drop = nn.Dropout(0.5)

        self.fc = nn.Linear(
            64,
            num_classes
        )

    def forward(
        self,
        x,
        lengths
    ):

        B, C, T, V = x.shape

        x = (
            x.permute(
                0,
                2,
                3,
                1
            )
            .contiguous()
            .view(
                B,
                T,
                V * C
            )
        )

        x = self.data_norm(x)

        packed_x = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, h_n = self.gru(
            packed_x
        )

        out = self.drop(
            h_n[-1]
        )

        return self.fc(out)