import torch 
from torch import nn

class DigitModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn_block = nn.Sequential(
          
            nn.Conv2d(in_channels=1, out_channels=128,kernel_size=(3,3),stride=4, padding = 1), 
         
         
            nn.BatchNorm2d(128),
       
            nn.ReLU(),
         
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3),stride=4, padding = 1),
        
            nn.BatchNorm2d(256),
       
            nn.ReLU(),
          
            nn.MaxPool2d(kernel_size=(2,2)),
        
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(3,3), stride=4,padding = 1),
          
            nn.BatchNorm2d(64),
         
            nn.ReLU(),
      
            nn.MaxPool2d(kernel_size=(2,2))

        )

        self.linear_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(256, 2)
            
        )

    def forward(self, images):
        x = self.cnn_block(images)
        logits = self.linear_block(x)

        return logits
