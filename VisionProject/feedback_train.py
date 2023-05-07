import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.models import resnet50
from torchvision.models import resnet18
# from mobilenet import mobilenetv3_large
from feedbacknet import EventDetector


# class CustomDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.img_paths = os.listdir(root_dir)
#
#     def __len__(self):
#         return len(self.img_paths)
#
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.root_dir, self.img_paths[idx])
#         image = Image.open(img_path)
#         if self.transform:
#             image = self.transform(image)
#         label = 1 if idx <= 498 else 0
#         return image, label

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = sorted(os.listdir(root_dir)) # 파일 이름을 정렬

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_paths[idx])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        label = 1 if int(self.img_paths[idx][:-4]) <= 498 else 0


# 디바이스 설정
if __name__ == '__main__':
    device = torch.device("cuda")

# 하이퍼파라미터 설정
    lr = 0.001
    batch_size = 16 #16
    num_epochs = 10

# 데이터 전처리
    transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

# 데이터셋 불러오기
    train_dataset = CustomDataset('./data/test0', transform=transform)

# 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# 모델 정의

    model = EventDetector().to(device)

# 손실함수 및 최적화 알고리즘 설정
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

# 학습 시작
    print("check")
    for epoch in range(num_epochs+1):
        running_loss = 0.0
        for inputs, labels in tqdm(list(train_loader)):
        # 입력 데이터와 레이블을 디바이스에 할당
            inputs = inputs.to(device)
            labels = labels.float().to(device)
        # 경사 초기화 및 순전파
            optimizer.zero_grad()
            outputs = model(inputs)
        # 손실 계산 및 역전파
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
        # 러닝 로스 업데이트
            running_loss += loss.item()
        if epoch % 10 == 0:
            torch.save({'optimizer_state_dict': optimizer.state_dict(),'model_state_dict': model.state_dict()}, 'feedbackmodel/feed_{}.pth.tar'.format(epoch))

#torch.save(model.state_dict(), 'feed.pth.tar')
    # epoch마다 Loss계산
