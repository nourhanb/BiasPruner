from torchvision import transforms, models


dataset = 'xray'
num_classes = 19
batch = 100
train_epochs = 20

finetune_epochs = 5


weights_original = '/home/jfayyad/PycharmProjects/BiasWashV2/weights/Original_model/Original_model.pth'
pruned_path = '/home/jfayyad/PycharmProjects/BiasWashV2/weights/Pruned_model'
weights_finetuned = "/home/jfayyad/PycharmProjects/BiasWashV2/weights/Tuned_model/Tuned_model2.pth"


NIH_base_meta = "/home/jfayyad/PycharmProjects/BiasWashV2/Bias_Exp/tasks/task1"
NIH_root = '/home/jfayyad/PycharmProjects/prune4fair/Data_splits/NIH XRAY/images/'
NIH_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])


pruning_ratio = 0.5
