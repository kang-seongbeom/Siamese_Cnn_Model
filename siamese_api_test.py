import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request
import json
import io


class Config():
    # find =_src, comapare_src와 경로 맞춰야 함
    training_dir = ".\images1\Training"
    testing_dir = ".\images1\Testing"
    path = "./siamese_cnn_model_cpu.pt"
    get_fine_key = "file_url"

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class GetTwoData(Dataset):
    def __init__(self, image_tuple, compare_img, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert
        self.image_tuple = image_tuple
        self.compare_img = compare_img

    def __getitem__(self, index):
        img0_tuple = self.image_tuple
        img1_tuple = self.compare_img

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/photo', methods=['POST'])
def photo():
    img_byte = request.files[Config.get_fine_key].read()
    data_io = io.BytesIO(img_byte)
    img = Image.open(data_io)
    img_type = str(type(img))

    return img_type

def get_image():
    img_byte = request.files[Config.get_fine_key].read()
    data_io = io.BytesIO(img_byte)
    return data_io

@app.route('/siamese', methods=['POST'])
def siamese():
    result_dict = dict()

    # spring server로 부터 받은 이미지를 튜플로 전환
    image_tuple = (get_image(), 0)

    # 테스트 이미지 파일에서 이미지 가져오기
    folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)

    # 비교할 이미지 개수 만큼 반복해 비교할 이미지와 1:1 비교할 수 있도록 함
    for i in range(1, len(folder_dataset_test) + 1):
        compare_src = ".\images1\Testing\s" + str((i)) + '\\' + str((i)) + ".jpg"
        compare_test = dset.ImageFolder(root=Config.testing_dir)
        for value in compare_test.imgs:
            if (value[0] == compare_src):
                compare_tuple = value
                break

        siamese_dataset = GetTwoData(image_tuple, compare_tuple, imageFolderDataset=folder_dataset_test,
                                     transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                   transforms.ToTensor()
                                                                   ])
                                     , should_invert=False)

        test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=True)
        dataiter = iter(test_dataloader)
        x0, _, _ = next(dataiter)

        _, x1, label2 = next(dataiter)
        concatenated = torch.cat((x0, x1), 0)
        output1, output2 = model(Variable(x0), Variable(x1))
        euclidean_distance = F.pairwise_distance(output1, output2)
        # imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
        result_dict[compare_src] = euclidean_distance.item()

    # 결과 값
    # print(result_dict)

    # 정렬
    sorted_result = sorted(result_dict.items(), key=lambda item: item[1])

    # 정렬을 하면 list로 바뀌므로 다시 dictionary로 변환
    dict_ = dict(sorted_result)

    # dictionary를 json으로 변환
    json_result = json.dumps(dict_)
    return json_result

if __name__ == "__main__":
    model = torch.load(Config.path)
    app.run(debug=False, host="127.0.0.1", port=5000)
