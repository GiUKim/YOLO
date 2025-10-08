import torch
import torch.nn as nn
import torch.nn.functional as f
from config import DATASET_CONFIG

class YOLOv1Backbone(nn.Module):
    def __init__(self, ch=3, num_classes=80, use_batch_norm=False):
        super().__init__()
        self.ch = ch
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        
        # Helper function to add BN after conv if enabled
        def conv_with_bn(in_channels, out_channels, kernel_size, stride=1, padding=0):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
            if self.use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.1))
            return nn.Sequential(*layers)
        
        # Layer 1
        self.conv1_1 = conv_with_bn(DATASET_CONFIG['input_channels'], 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2_1 = conv_with_bn(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 3
        self.conv3_1 = conv_with_bn(192, 128, kernel_size=1, stride=1, padding=0)
        self.conv3_2 = conv_with_bn(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = conv_with_bn(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv3_4 = conv_with_bn(256, 512, kernel_size=3, stride=1, padding=1)
        self.maxpool3_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 4
        self.conv4_1 = conv_with_bn(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv4_2 = conv_with_bn(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = conv_with_bn(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv4_4 = conv_with_bn(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_5 = conv_with_bn(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv4_6 = conv_with_bn(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_7 = conv_with_bn(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv4_8 = conv_with_bn(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_9 = conv_with_bn(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv4_10 = conv_with_bn(512, 1024, kernel_size=3, stride=1, padding=1)
        self.maxpool4_10 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 5
        self.conv5_1 = conv_with_bn(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv5_2 = conv_with_bn(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = conv_with_bn(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv5_4 = conv_with_bn(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv5_5 = conv_with_bn(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.conv5_6 = conv_with_bn(1024, 1024, kernel_size=3, stride=2, padding=1)

        # Layer 6
        self.conv6_1 = conv_with_bn(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = conv_with_bn(1024, 1024, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(1024 * 7 * 7, 4096)
        self.relu_fc1 = nn.LeakyReLU(0.1)
        self.dropout_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(4096, 7 * 7 * (5 * 2 + self.num_classes))
    
    def forward(self, x):
        # Conv layers (now using Sequential modules with optional BN)
        x = self.conv1_1(x)
        x = self.maxpool1_1(x)
        
        x = self.conv2_1(x)
        x = self.maxpool2_1(x)
        
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.maxpool3_4(x)
        
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = self.conv4_6(x)
        x = self.conv4_7(x)
        x = self.conv4_8(x)
        x = self.conv4_9(x)
        x = self.conv4_10(x)
        x = self.maxpool4_10(x)
        
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        x = self.conv5_5(x)
        x = self.conv5_6(x)
        
        x = self.conv6_1(x)
        x = self.conv6_2(x)

        # Flatten for FC layers
        x = x.view(x.size(0), -1)  # (batch_size, 7*7*1024)
        
        # FC layers
        x = self.relu_fc1(self.fc1(x))
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        
        return x

class YOLOv1(nn.Module):
    def __init__(self, ch=3, num_classes=80, use_batch_norm=False):
        super(YOLOv1, self).__init__()
        self.ch = ch
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        self.backbone = YOLOv1Backbone(ch=self.ch, num_classes=self.num_classes, use_batch_norm=self.use_batch_norm)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)  # (batch_size, 7*7*(5*2+num_classes))
        x = x.view(x.size(0), 7, 7, 5 * 2 + self.num_classes)  # (batch_size, 7, 7, 5*2+num_classes)
        bbox_output = torch.sigmoid(x[:, :, :, :5*2])
        class_output = torch.softmax(x[:, :, :, 5*2:], dim=-1)
        x = torch.cat([bbox_output, class_output], dim=-1)
        return x


if __name__ == "__main__":
    model = YOLOv1(ch=3, num_classes=80)
    
    x = torch.randn(1, 3, 448, 448)
    
    print("=" * 50)
    print("YOLOv1 Model Test")
    print("=" * 50)
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (1, 7, 7, 90) for 80 classes")
    print("=" * 50)
    
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
