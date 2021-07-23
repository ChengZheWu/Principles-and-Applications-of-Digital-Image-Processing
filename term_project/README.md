# Face Mask Detection using Deep Learning Method
## ABSTRACT
近期，新冠病毒蔓延全世界，造成了很嚴重的危機，為了防止人與人之間的接觸造成傳染擴大，只要是人群聚集地、室內空間都需要強制戴上口罩，減少病毒傳染的機率，防止疫情的增長，因此，部分特地區域需要
檢測人們進出是否佩戴口罩。在本實驗中，使用的訓練資料集為kaggle中的Face Mask Detection資料集，利用deep learning來偵測人們有無配戴口罩，使用最先進的檢測模型之一"YOLOv3"，且以transfer learning的方式進行finetune training，最終得到mAP為77.86%。
## Results
Label                 | AP       
:---------------------|----------:
with mask             |81.43     
without mask          |90.44  
mask weared incorrect |71.72
**mAP**               |**77.89**
