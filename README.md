# Pytorch-Lightning-practice-repository
## [Pytorch lightning](https://wikidocs.net/156985)
> 참가자 모집과제 수행 
> 출처 및 참고
## 설치
>pytorch_lighting install 

⚠️ 주의 1 : 지금 쓰고있는 python == 3.10.4 ver에서는 다음과 같은 Issue가 발생하는 것 같다. 
![image](https://user-images.githubusercontent.com/45085563/174079754-6b0b783c-d85e-4e50-ba9e-c9df2d8e7018.png)

>![image](https://user-images.githubusercontent.com/45085563/174085422-c8495785-7a69-4d6c-841a-da3c3dbdc81a.png)
찾아보니 애초에 3.9까지 밖에 지원을 안한다... 가상환경을 새로 설치하자 다음부터는 [공식 문서를 참조하자...](https://github.com/Borda/pyDeprecate/issues/18)

### 가상환경 생성 후 필요한 파일 install 
가상환경을 새로 생성하고(3.9) 커널도 생성하자 
[블로그 참조](https://velog.io/@reversesky/%EA%B0%80%EC%83%81%ED%99%98%EA%B2%BD-%EC%B4%88%EA%B8%B0-%EC%84%B8%ED%8C%85)
```
conda create --name lightning python=3.9
```

### 자주 쓰는것들 install
```
conda install ipykernel -y
ipython kernel install --user --name=[lightning]
conda install jupyter -y 
conda install jupyterlab -y
conda install -c anaconda numpy -y
conda install -c anaconda pandas -y
conda install -c conda-forge matplotlib -y
conda install -c anaconda scikit-learn -y
conda install -c conda-forge opencv -y
conda install -c conda-forge tqdm -y
conda install pytorch-lightning -c conda-forge -y
conda install -c conda-forge jupyterlab_widgets -y
conda install -c conda-forge ipywidgets-y
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
![image](https://user-images.githubusercontent.com/45085563/174089748-e14ead53-5a4d-4593-b54c-bb95bb3a09c6.png)
> 드디어 error없이 깔끔하게 된다! 

----------
 
⚠️ 주의 2
```
from pytorch_lightning import LightningDataModule
```
가 안된다. python = 3.8.10으로 해야할 것 같다. [안됨...](https://github.com/Lightning-AI/lightning/issues/12784)

### 일단 코랩에서 진행 
##  
```python
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
```
필요한 파일을 import 

```python
class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10) #linear를 생성 

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb): #정의 구간? 
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02) #optimizer를 설정하는 곳인가 보다. 
```
training부분 마치 keras나 sklearn같다. 
```python
mnist_model = MNISTModel()

# Init DataLoader from MNIST Dataset
train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

# Initialize a trainer
trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=3,
    progress_bar_refresh_rate=20,
)

# Train the model ⚡
trainer.fit(mnist_model, train_loader)
```


### 결과
![image](https://user-images.githubusercontent.com/45085563/174110663-a7512e76-c29b-4f32-9920-645222396ca1.png)
epoch이 3일때의 결과 

epoch 20? 
![image](https://user-images.githubusercontent.com/45085563/174112543-4c808751-7583-465b-a13e-b3ea70d640dc.png)
linear가 너무 적었나보다.
