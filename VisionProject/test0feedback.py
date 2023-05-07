import torch
import torchvision.transforms as transforms
from PIL import Image
from feedbacknet import EventDetector

# 모델 불러오기
model = EventDetector()
save_dict = torch.load('feedbackmodel/feed_10.pth.tar')
model.load_state_dict(save_dict['model_state_dict'])
#save_dict = torch.load('feed.pth.tar')
#model.load_state_dict(save_dict)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
with torch.no_grad():
    for i in range(0, 500):
        img = Image.open('./data/test0/{}.png'.format(i))
        img = transform(img)
        img = img.unsqueeze(0)
        output = model(img.to(device))
        print(output)