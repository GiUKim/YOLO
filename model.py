import torch
import torch.nn as nn
import torch.nn.functional as f

class YOLOv1Backbone(nn.Module):
    def __init__(self, ch=3, num_classes=80):
        super().__init__()
        self.ch = ch
        self.num_classes = num_classes
        self.conv1_1 = nn.Conv2d(3 if self.ch == 3 else 1, 64, kernel_size=7, stride=2, padding=3)
        self.relu1_1 = nn.LeakyReLU(0.1)
        self.maxpool1_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.relu2_1 = nn.LeakyReLU(0.1)
        self.maxpool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0)
        self.relu3_1 = nn.LeakyReLU(0.1)
        self.conv3_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_2 = nn.LeakyReLU(0.1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.relu3_3 = nn.LeakyReLU(0.1)
        self.conv3_4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu3_4 = nn.LeakyReLU(0.1)
        self.maxpool3_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.relu4_1 = nn.LeakyReLU(0.1)
        self.conv4_2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_2 = nn.LeakyReLU(0.1)
        self.conv4_3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.relu4_3 = nn.LeakyReLU(0.1)
        self.conv4_4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_4 = nn.LeakyReLU(0.1)
        self.conv4_5 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.relu4_5 = nn.LeakyReLU(0.1)
        self.conv4_6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_6 = nn.LeakyReLU(0.1)
        self.conv4_7 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.relu4_7 = nn.LeakyReLU(0.1)
        self.conv4_8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_8 = nn.LeakyReLU(0.1)
        self.conv4_9 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.relu4_9 = nn.LeakyReLU(0.1)
        self.conv4_10 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.relu4_10 = nn.LeakyReLU(0.1)
        self.maxpool4_10 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.relu5_1 = nn.LeakyReLU(0.1)
        self.conv5_2 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.relu5_2 = nn.LeakyReLU(0.1)
        self.conv5_3 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.relu5_3 = nn.LeakyReLU(0.1)
        self.conv5_4 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.relu5_4 = nn.LeakyReLU(0.1)
        self.conv5_5 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.relu5_5 = nn.LeakyReLU(0.1)
        self.conv5_6 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)
        self.relu5_6 = nn.LeakyReLU(0.1)

        self.conv6_1 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.relu6_1 = nn.LeakyReLU(0.1)
        self.conv6_2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.relu6_2 = nn.LeakyReLU(0.1)

        self.fc1 = nn.Linear(1024 * 7 * 7, 4096)
        self.relu_fc1 = nn.LeakyReLU(0.1)
        self.dropout_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(4096, 7 * 7 * (5 * 2 + self.num_classes))
    
    def forward(self, x):
        # Conv layers
        x = self.relu1_1(self.conv1_1(x))
        x = self.maxpool1_1(x)
        print(f"Conv1_1: {x.shape}")
        
        x = self.relu2_1(self.conv2_1(x))
        x = self.maxpool2_1(x)
        print(f"Conv2_1: {x.shape}")
        
        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.relu3_3(self.conv3_3(x))
        x = self.relu3_4(self.conv3_4(x))
        x = self.maxpool3_4(x)
        print(f"Conv3_4: {x.shape}")
        
        x = self.relu4_1(self.conv4_1(x))
        x = self.relu4_2(self.conv4_2(x))
        x = self.relu4_3(self.conv4_3(x))
        x = self.relu4_4(self.conv4_4(x))
        x = self.relu4_5(self.conv4_5(x))
        x = self.relu4_6(self.conv4_6(x))
        x = self.relu4_7(self.conv4_7(x))
        x = self.relu4_8(self.conv4_8(x))
        x = self.relu4_9(self.conv4_9(x))
        x = self.relu4_10(self.conv4_10(x))
        x = self.maxpool4_10(x)
        print(f"Conv4_10: {x.shape}")
        
        x = self.relu5_1(self.conv5_1(x))
        x = self.relu5_2(self.conv5_2(x))
        x = self.relu5_3(self.conv5_3(x))
        x = self.relu5_4(self.conv5_4(x))
        x = self.relu5_5(self.conv5_5(x))
        x = self.relu5_6(self.conv5_6(x))
        print(f"Conv5_6: {x.shape}")
        
        x = self.relu6_1(self.conv6_1(x))
        x = self.relu6_2(self.conv6_2(x))
        print(f"Conv6_2: {x.shape}")

        # Flatten for FC layers
        x = x.view(x.size(0), -1)  # (batch_size, 7*7*1024)
        print(f"Flatten: {x.shape}")
        
        # FC layers
        x = self.relu_fc1(self.fc1(x))
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        print(f"FC2: {x.shape}")
        
        return x

class YOLOv1(nn.Module):
    def __init__(self, ch=3, num_classes=80):
        super(YOLOv1, self).__init__()
        self.ch = ch
        self.num_classes = num_classes
        self.backbone = YOLOv1Backbone(ch=self.ch, num_classes=self.num_classes)

    def forward(self, x):
        x = self.backbone(x)  # (batch_size, 7*7*(5*2+num_classes))
        x = x.view(x.size(0), 7, 7, 5 * 2 + self.num_classes)  # (batch_size, 7, 7, 5*2+num_classes)
        return x


if __name__ == "__main__":
    # 모델 생성
    model = YOLOv1(ch=3, num_classes=80)
    
    # 샘플 입력 텐서 생성
    x = torch.randn(1, 3, 448, 448)
    
    print("=" * 50)
    print("YOLOv1 Model Test")
    print("=" * 50)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (1, 7, 7, 90) for 80 classes")
    print("=" * 50)
    
    # torchsummary로 모델 구조 확인
    try:
        from torchsummary import summary
        print("\nModel Summary:")
        print("-" * 50)
        model = model.to("cuda")
        summary(model, (3, 448, 448))
    except ImportError:
        print("\nNote: torchsummary not installed. Install with: pip install torchsummary")
        print("Model parameters:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
