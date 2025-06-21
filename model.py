import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, num_classes=10, img_shape=(1, 28, 28)):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        input_dim = z_dim + num_classes
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat((z, label_embedding), dim=1)
        img = self.model(x)
        return img.view(img.size(0), *self.img_shape)
