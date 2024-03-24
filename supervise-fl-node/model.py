import torch
import torch.nn as nn
from tdnn import TDNN


## audio input: [bsz, 20, 87]
class audio_encoder(nn.Module):
    """
    model for audio data
    """

    def __init__(self):
        super().__init__()

        self.tdnn1 = TDNN(input_dim=20, output_dim=256, context_size=5, dilation=5)
        self.tdnn2 = TDNN(input_dim=256, output_dim=512, context_size=5, dilation=5)
        self.tdnn3 = TDNN(input_dim=512, output_dim=256, context_size=5, dilation=5)
        self.tdnn4 = TDNN(input_dim=256, output_dim=128, context_size=3, dilation=3)
        self.tdnn5 = TDNN(input_dim=128, output_dim=128, context_size=3, dilation=3)

        self.gru = nn.GRU(128, 16, 2, batch_first=True)

    def forward(self, x):

        self.gru.flatten_parameters()

        x = x.transpose(1,2)

        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)
        
        # print("original audio feature:", x.shape)#[8, 15, 128]

        x = x.reshape(x.size(0), -1, 128)#[bsz, 15, 128]
        x, _ = self.gru(x)#.flatten_parameters()

        # print("audio feature after gru:", x.shape)#[bsz, 15, 16]

        out = x.reshape(x.size(0), -1)#[bsz, 240]

        return out


## depth input: [bsz, 1, 16, 112, 112]
class depth_encoder(nn.Module):
    """
    model for depth video
    """

    def __init__(self):
        super().__init__()

        # conv1 input (n*1*16*112*112), conv5 output (n*512*1*4*4)
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU(),
        )
        self.conv3a = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            # nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU(),
        )
        self.conv3b = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU(),
        )
        self.conv4a = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            # nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU(),
        )
        self.conv4b = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU(),
        )
        self.conv5a = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
        )
        self.conv5b = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU(),
        )

        self.gru = nn.GRU(288, 16, 2, batch_first=True)

    def forward(self, x):

        self.gru.flatten_parameters()

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.conv5a(x)
        x = self.conv5b(x)

        # print("original depth feature:", x.shape)#[bsz, 64, 1, 4, 4]

        x = x.view(x.size(0), 16, -1)#[bsz, 16, 64]
        x, _ = self.gru(x)

        out = x.reshape(x.size(0), -1)#[bsz, 256]

        # print("depth feature after gru:", out.shape)

        return out


## radar input: [bsz, 20, 2, 16, 32, 16]
class radar_encoder(nn.Module):
    """
    For radar: input size (20*16*32*16)
    """

    def __init__(self):
        super().__init__()

        # conv1 input (n*20)*2*16*32*16, conv4 output (n*20)*256*2*4*2
        self.conv1 = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU(),
        )
        self.conv3a = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU(),
        )
        # self.conv3b = nn.Sequential(
        #     nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        #     nn.BatchNorm3d(256),
        #     nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        #     nn.ReLU(),
        # )
        self.conv4a = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU(),
        )
        # self.conv4b = nn.Sequential(
        #     nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        #     nn.BatchNorm3d(512),
        #     nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        #     nn.ReLU(),
        # )
        self.conv5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            # nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=1024, hidden_size=16, num_layers=2, bidirectional=False, batch_first=True)



    def forward(self, x):

        self.lstm.flatten_parameters()

        bsz = x.size(0)
        x = x.view(-1, 2, 16, 32, 16)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3a(x)
        # x = self.conv3b(x)
        x = self.conv4a(x)
        # x = self.conv4b(x)
        x = self.conv5(x)

        # print("original radar feature:", x.shape)#[160, 64, 2, 4, 2]
        x = x.view(bsz, 20, -1)  # [bsz, 20, 1024]

        out, _ = self.lstm(x)#.flatten_parameters()  # [bsz, 20, 32]
        # print("radar feature after lstm:", out.shape)# [bsz, 20, 16]

        out = out.reshape(out.size(0), -1)#[bsz, 320]

        return out


class MySingleModel(nn.Module):

    def __init__(self, num_classes, modality):
        super().__init__()

        if modality == 'audio':#[1498907]
            self.encoder = audio_encoder()#[1496256]
            self.classifier = nn.Sequential(
                nn.Linear(240, num_classes),
                nn.Softmax()
                )#[2651]
        elif modality == 'depth':#[2223883]
            self.encoder = depth_encoder()#[2221056]
            self.classifier = nn.Sequential(
                nn.Linear(256, num_classes),
                nn.Softmax()
                )#[2827]        
        elif modality == 'radar':#[629771]
            self.encoder = radar_encoder()#[626240]
            self.classifier = nn.Sequential(
            nn.Linear(320, num_classes),
            nn.Softmax()
            )#[3531]

    def forward(self, x):
        # print(x.shape)
        feature = self.encoder(x)
        output = self.classifier(feature)

        return output


class Encoder3(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_1 = audio_encoder()
        self.encoder_2 = depth_encoder()
        self.encoder_3 = radar_encoder()

    def forward(self, x1, x2, x3):

        feature_1 = self.encoder_1(x1)
        feature_2 = self.encoder_2(x2)
        feature_3 = self.encoder_3(x3)

        return feature_1, feature_2, feature_3


class My3Model(nn.Module):

    def __init__(self, num_classes):#[4352539]
        super().__init__()

        self.encoder = Encoder3()#[4343552]

        self.classifier = nn.Sequential(
        nn.Linear(816, 512),
        nn.Linear(512, 512),
        nn.Linear(512, num_classes),
        nn.Softmax(dim=1)
        )#[8987]


    def forward(self, x1, x2, x3):

        feature_1, feature_2, feature_3 = self.encoder(x1, x2, x3)
        feature = torch.cat((feature_1, feature_2, feature_3), dim=1)

        output = self.classifier(feature)


        return output


class My3Model_unsupervise(nn.Module):

    def __init__(self):#[4352539]
        super().__init__()

        self.encoder = Encoder3()#[4343552]

        self.head_1 = nn.Sequential(
            nn.Linear(240, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )

        self.head_2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )

        self.head_3 = nn.Sequential(
            nn.Linear(320, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )
     
    def forward(self, x1, x2, x3):

        feature_1, feature_2, feature_3 = self.encoder(x1, x2, x3)

        f1 = self.head_1(feature_1)
        f2 = self.head_2(feature_2)
        f3 = self.head_3(feature_3)

        return f1, f2, f3

