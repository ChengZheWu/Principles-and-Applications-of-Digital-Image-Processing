# Face Mask Detection using Transfer Learning of YOLOv3

## Abstract
近期，新冠病毒蔓延全世界，造成了很嚴重的危機，為了防止人與人之間的接觸造成傳染擴大，只要是人群聚集地、室內空間都需要強制戴上口罩，減少病毒傳染的機率，防止疫情的增長，因此許多地方需要檢測人們進出是否佩戴口罩，為了讓檢測更方便快速且準確，越來越多人投入這方面的研究主題。在本實驗中，使用的訓練資料集為kaggle中的Face Mask Detection資料集，利用deep learning來偵測人們有無配戴口罩，使用最先進的檢測模型之一"YOLOv3"，且以transfer learning的方式進行finetune training，最終得到mAP為77.86%。

## Dataset
本實驗使用Kaggle的[Face Mask Detection Dataset](https://www.kaggle.com/andrewmvd/face-mask-detection)，總共有853張影像，分成三個類別，分別為with mask、without mask和mask weared incorrect。

## Method
使用Transfer Learning方法進行訓練  
**Model:** YOLOv3 (Darknet 53)  
**Loss Function:** YOLO Loss  
**Optimizer:** Adam  
### Training tricks
Training set : Validation set : Testing set = 8 : 1 : 1  
**First stage:** batch size=32 , epochs=50, initial learning rate = 1e-3, 將除了output layer以外的layers都先freeze來進行training。  
**Second Stage:** batch size=4, epochs=50, initial learning rate = 1e-4, 接續前面的訓練結果，此時將所有的layers設定成learnable後進行訓練，訓練中會監測val loss的狀況來調整learning rate，設定為連續3 epochs都沒進步就將learning rate乘上0.1，並且使用early stopping機制，設定為連續10 epochs都沒再進步就停止訓練，最後停止在90 epochs。  

## Results

#### Loss with training and validation
![YOLO Loss](https://github.com/ChengZheWu/Principles-and-Applications-of-Digital-Image-Processing/blob/main/term_project/data/loss.png)

#### Perfromance in testing set
Label                 | AP       
:---------------------|----------:
with mask             |81.43     
without mask          |80.44  
mask weared incorrect |71.72
**mAP**               |**77.89**

#### Examples of the prediction
input                 | Prediction     
:---------------------|:----------
![809](https://github.com/ChengZheWu/Principles-and-Applications-of-Digital-Image-Processing/blob/main/term_project/data/maksssksksss809.png)|![809p](https://github.com/ChengZheWu/Principles-and-Applications-of-Digital-Image-Processing/blob/main/term_project/data/809.png)
![832](https://github.com/ChengZheWu/Principles-and-Applications-of-Digital-Image-Processing/blob/main/term_project/data/maksssksksss832.png)|![832p](https://github.com/ChengZheWu/Principles-and-Applications-of-Digital-Image-Processing/blob/main/term_project/data/832.png)
![807](https://github.com/ChengZheWu/Principles-and-Applications-of-Digital-Image-Processing/blob/main/term_project/data/maksssksksss807.png)|![807p](https://github.com/ChengZheWu/Principles-and-Applications-of-Digital-Image-Processing/blob/main/term_project/data/807.png)
![795](https://github.com/ChengZheWu/Principles-and-Applications-of-Digital-Image-Processing/blob/main/term_project/data/maksssksksss795.png)|![795p](https://github.com/ChengZheWu/Principles-and-Applications-of-Digital-Image-Processing/blob/main/term_project/data/795.png)

## Reference
[Face Mask Detection Dataset, online accessed Dec 30, 2020](https://www.kaggle.com/andrewmvd/face-mask-detection)  
[Face Mask Detection using Transfer Learning of InceptionV3, 2020](https://arxiv.org/abs/2009.08369)  
[YOLOv3: An Incremental Improvement, 2018](https://arxiv.org/abs/1804.02767)  
[keras-yolo3](https://github.com/qqwweee/keras-yolo3)  
[YOLO3 (Detection, Training, and Evaluation)](https://github.com/experiencor/keras-yolo3)  
