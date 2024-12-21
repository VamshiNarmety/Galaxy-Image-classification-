import torch.nn as nn
import timm
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class EfficientNetB1(nn.Module):
    def __init__(self, num_classes=3):
        super(EfficientNetB1, self).__init__()
        self.pretrained_model = timm.create_model('efficientnet_b1', pretrained=True)
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


def EfficientNet_b1(num_classes):
    model = EfficientNetB1(num_classes=num_classes).to(device)
    return model
