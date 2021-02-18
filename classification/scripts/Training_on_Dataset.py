import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
# from src.run_model import run_model, _train
import time

from read_data import ReadDataset
# from data.dogs import DogsDataset
# from src.models import Dog_Classifier_FC

# data transformation to normalize data
# normalizing data helps your optimization algorithm converge more quickly

mean = (0.485, 0.456, 0.406)    # from transforms.Compose example (https://pytorch.org/docs/stable/torchvision/transforms.html)
std = (0.229, 0.224, 0.225)     # -"-
transform = transforms.Compose([transforms.Resize((32, 32)),
                                # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=mean, std=std)])

# training set split
training_dataset_size = [None]

running_time_list = []
train_accuracy_list = []
valid_accuracy_list = []
train_loss_list = []
valid_loss_list = []

for i in training_dataset_size:
    train_dataset = ReadDataset(trainset_name, transform=transform)
    valid_dataset = ReadDataset(validset_name, transform=transform)
    test_dataset = ReadDataset(testset_name, transform=transform)

    classes = ['samoyed', 'miniature_poodle', 'golden_retriever', 'great_dane',
               'dalmatian', 'collie', 'siberian_husky', 'yorkshire_terrier', 'chihuahua', 'saint_bernard']

    def imshow(img):
        img = img / 2 + 0.5  # have to unnormalize the images
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig("Dogs_imgs.png")
        plt.show()
    
    
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(10)))

    model = Dog_Classifier_FC()
    learning_rate = 1e-5
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    device = torch.device('cpu')

    start_time = time.time()
    model, train_loss, train_accuracy = run_model(model, running_mode='train', train_set=train_loader,
                                                  valid_set=valid_dataset,
                                                  test_set=test_dataset,
                                                  batch_size=10, learning_rate=learning_rate, n_epochs=100,
                                                  stop_thr=1e-4,
                                                  shuffle=True)
    end_time = time.time()
    running_time = end_time - start_time
    running_time_list.append(running_time)

    train_loss_list.append(np.mean(train_loss['train']))
    valid_loss_list.append(np.mean(train_loss['valid']))
    train_accuracy_list.append(np.mean(train_accuracy['train']))
    valid_accuracy_list.append(np.mean(train_accuracy['valid']))

    test_loss, test_accuracy = run_model(model, running_mode='test', train_set=train_loader,
                                         valid_set=valid_dataset,
                                         test_set=test_dataset,
                                         batch_size=10, learning_rate=learning_rate, n_epochs=100, stop_thr=1e-4,
                                         shuffle=True)

plt.figure()
plt.plot(range(len(train_loss['train'])), train_loss['train'], label='Training Loss')
plt.plot(range(len(train_loss['valid'])), train_loss['valid'], label='Validation Loss')
plt.title('Training and Validation Loss Vs. Epoch Number')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.legend(loc="best")
plt.savefig("Q3-loss.png")
plt.show()

plt.figure()
plt.plot(range(len(train_accuracy['train'])), train_accuracy['train'], label='Training Accuracy')
plt.plot(range(len(train_accuracy['valid'])), train_accuracy['valid'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy Vs. Epoch Number')
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.legend(loc="best")
plt.savefig("Q3-accuracy.png")
plt.show()

print("The accuracy of your model on the testing set:", test_accuracy)
