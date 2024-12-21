import torch.nn as nn
import timm
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DenseNet121(nn.Module):
    def __init__(self, num_classes=3):
        super(DenseNet121, self).__init__()
        self.pretrained_model = timm.create_model('densenet121', pretrained=True)
        in_features = self.pretrained_model.classifier.in_features
        self.pretrained_model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.pretrained_model(x)
        return x


def DenseNet_121(num_classes, device='cpu'):
    model = DenseNet121(num_classes=num_classes).to(device)
    return model
