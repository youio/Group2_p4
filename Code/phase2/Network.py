import torch
import torch.nn as nn
import numpy as np

class IOmodel(nn.Module):
    def __init__(self, input_size=6, seq_len=10, linear_size=32, hidden_size=64, lstm_layers=1, dropout=0.3):
        super(IOmodel, self).__init__()

        self.linear = nn.Linear(input_size, linear_size)
        self.norm = nn.LayerNorm(linear_size)

        self.lstm_pos = nn.LSTM(input_size=linear_size,
                                hidden_size=hidden_size,
                                num_layers=lstm_layers,
                                dropout=dropout if lstm_layers > 1 else 0.0,
                                batch_first=True)

        self.lstm_ori = nn.LSTM(input_size=linear_size,
                                hidden_size=hidden_size,
                                num_layers=lstm_layers,
                                dropout=dropout if lstm_layers > 1 else 0.0,
                                batch_first=True)

        # Projection if dimensions mismatch for residual
        self.proj_residual = (linear_size != hidden_size)
        if self.proj_residual:
            self.res_proj = nn.Linear(linear_size, hidden_size)

        self.fc_pos = nn.Linear(hidden_size, 3)
        self.fc_ori = nn.Linear(hidden_size, 3)

    def forward(self, images, x):
        # x: (batch_size, seq_len, input_size)
        x = self.linear(x)                # -> (batch, seq, linear_size)
        x = self.norm(x)                  # layer normalization

        residual = x[:, -1, :]            # last timestep from linear path
        if self.proj_residual:
            residual = self.res_proj(residual)

        out_pos, _ = self.lstm_pos(x)
        out_ori, _ = self.lstm_ori(x)

        pos = self.fc_pos(out_pos[:, -1, :] + residual)
        ori = self.fc_ori(out_ori[:, -1, :] + residual)

        # print('pos,ori', pos, ori)
        return pos ,ori
    
import torch
import torch.nn as nn

class VOmodel(nn.Module):
    def __init__(self, input_channels=6, hidden_size=64, linear_out=128, lstm_layers=1, dropout=0.3):
        super(VOmodel, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

        self.cnn = nn.Sequential(
            conv_block(input_channels, 32),     # 244 → 122
            conv_block(32, 64),                # 122 → 61
            conv_block(64, 128),               # 61 → 30
            conv_block(128, 128),              # 30 → 15
            conv_block(128, 128),              # 15 → 7
            conv_block(128, 512),              # 7 → 3
        )

        # Output from CNN: (batch, 512, 3, 3) → flatten to (batch, 4608)
        self.flat_size = 512 * 3 * 3
        self.linear = nn.Linear(self.flat_size, linear_out)
        self.norm = nn.LayerNorm(linear_out)

        self.lstm_pos = nn.LSTM(input_size=linear_out,
                                hidden_size=hidden_size,
                                num_layers=lstm_layers,
                                dropout=dropout if lstm_layers > 1 else 0.0,
                                batch_first=True)

        self.lstm_ori = nn.LSTM(input_size=linear_out,
                                hidden_size=hidden_size,
                                num_layers=lstm_layers,
                                dropout=dropout if lstm_layers > 1 else 0.0,
                                batch_first=True)

        self.fc_pos = nn.Linear(hidden_size, 3)
        self.fc_ori = nn.Linear(hidden_size, 3)

    def forward(self,images, x):
        # x: (batch, seq_len, channels=6, H=224, W=224)
        batch, seq_len, C, H, W = images.size()
        images = images.view(-1, C, H, W)                  # merge batch and seq for CNN
        images = self.cnn(images)                          # → (batch*seq, 512, 3, 3)
        images = images.view(batch, seq_len, -1)           # → (batch, seq, 4608)
        images = self.norm(self.linear(images))            # → (batch, seq, linear_out)

        out_pos, _ = self.lstm_pos(images)
        out_ori, _ = self.lstm_ori(images)

        pos = self.fc_pos(out_pos.reshape(100, -1))
        ori = self.fc_ori(out_ori.reshape(100, -1))
        return pos, ori

        
class VIOmodel(nn.Module):
    def __init__(self, input_channels=6,seq_len=10,linear_size =32, hidden_size=64, linear_out=128, lstm_layers=1, dropout=0.3):
        super(VIOmodel, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

        self.cnn = nn.Sequential(
            conv_block(input_channels, 32),     # 244 → 122
            conv_block(32, 64),                # 122 → 61
            conv_block(64, 128),               # 61 → 30
            conv_block(128, 128),              # 30 → 15
            conv_block(128, 128),              # 15 → 7
            conv_block(128, 512),              # 7 → 3
        )

        self.linear = nn.Linear(input_channels, linear_size)
        self.norm = nn.LayerNorm(linear_size)

        self.flatsize = 512*3*3
        self.linear_vo = nn.Linear(self.flatsize, linear_out)
        self.norm_vo = nn.LayerNorm(linear_out)

        # Projection if dimensions mismatch for residual
        self.proj_residual = (linear_size != hidden_size)
        if self.proj_residual:
            self.res_proj = nn.Linear(linear_size, hidden_size)


        self.lstm_io = nn.LSTM(input_size=linear_size,
                                hidden_size=hidden_size,
                                num_layers=lstm_layers,
                                dropout=dropout if lstm_layers > 1 else 0.0,
                                batch_first=True)

        self.lstm_vo = nn.LSTM(input_size=linear_out,
                                hidden_size=hidden_size,
                                num_layers=lstm_layers,
                                dropout=dropout if lstm_layers > 1 else 0.0,
                                batch_first=True)
        
        
        self.fuse = nn.Linear(hidden_size*2, linear_size)

        self.lstm_pos = nn.LSTM(input_size=linear_size,
                                hidden_size=hidden_size,
                                num_layers=lstm_layers,
                                dropout=dropout if lstm_layers > 1 else 0.0,
                                batch_first=True)

        self.lstm_ori = nn.LSTM(input_size=linear_size,
                                hidden_size=hidden_size,
                                num_layers=lstm_layers,
                                dropout=dropout if lstm_layers > 1 else 0.0,
                                batch_first=True)
        
        self.fc_pos = nn.Linear(hidden_size, 3)
        self.fc_ori = nn.Linear(hidden_size, 3)

    def forward(self, images, x):

        # visual part
        batch, seq_len, C, H, W = images.size()
        images = images.view(-1, C, H, W)                  # merge batch and seq for CNN
        images = self.cnn(images)                          # → (batch*seq, 512, 3, 3)
        images = images.view(batch, seq_len, -1)           # → (batch, seq, 4608)
        images = self.norm_vo(self.linear_vo(images))            # → (batch, seq, linear_out)

        images, _ = self.lstm_vo(images)                      # (10, 10, hidden_size)


        # inertial part
        x = self.linear(x)                # -> (batch, seq, linear_size)
        x = self.norm(x)                  # layer normalization

        residual = x[:, -1, :]            # last timestep from linear path
        if self.proj_residual:
            residual = self.res_proj(residual)

        x, _ = self.lstm_io(x)
        x = (x[:, -1, :] + residual).squeeze(1)              # (100, hidden_size)

        features = torch.cat([images, x.view(10,10, -1)], dim=2)

        features= self.fuse(features)

        out_pos, _ = self.lstm_pos(features)
        out_ori, _ = self.lstm_ori(features)

        print(out_pos.shape)
        pos = self.fc_pos(out_pos.reshape(100, -1))
        ori = self.fc_ori(out_ori.reshape(100, -1))

        return pos, ori
    
class IMUNet(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, output_size=6, concat=False):
        super(IMUNet, self).__init__()
        self.hidden_size = hidden_size
        self.concat = concat

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)

        if self.concat:
            self.fc_concat = nn.Linear(hidden_size, 128)
        else:
            self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
            self.fc1 = nn.Linear(hidden_size, hidden_size)
            self.prelu = nn.PReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = out[:, -1, :]

        if self.concat:
            out = self.fc_concat(out)
        else:
            out1, _ = self.lstm1(x)
            out2, _ = self.lstm2(out1)
            out = out2[:, -1, :]
            out = self.fc1(out)
            out = self.prelu(out)
            out = self.fc2(out)

        return out