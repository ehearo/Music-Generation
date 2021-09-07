import os
import time
import copy
import numpy as np
import random
from ranger import Ranger
import torch
import torch.nn as nn
from sys import stdout
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms
import logging
from model import FacialEmotionEmbedding, DeepFaceEmotion
from utils import CreateDatasetFromDataFrame, weights_init
from config import mapping


FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(filename='training.log', level=logging.DEBUG, format=FORMAT)

worker_seed = 21
np.random.seed(worker_seed)
random.seed(worker_seed)
torch.manual_seed(worker_seed)
torch.cuda.manual_seed(worker_seed)
torch.cuda.manual_seed_all(worker_seed)

data_dir = 'C:/dataset/ready_to_use/train'
test_dir = 'C:/dataset/ready_to_use/valid'


def train(model_, device_, data_loader):
    global model_recorder
    since = time.time()
    model_.train()
    running_loss = 0.0
    running_corrects = 0
    sample_size = 0
    data_set_size = len(data_loader.dataset)

    for ind, data in enumerate(data_loader):
        loop_time_start = time.time()
        y_ = data[1].type(torch.LongTensor).to(device_)
        x_ = data[0].to(device_)

        # zero the parameter gradients
        optimizer.zero_grad()

        # get predict
        y_hat = model(x_)
        _, predicts = torch.max(y_hat, 1)

        # calculate loss
        loss = criterion(y_hat, y_)

        # back propagation
        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)

        # update optimizer
        optimizer.step()

        # statistics
        running_loss += loss.item() * x_.size(0)
        running_corrects += torch.sum(predicts == y_.data)

        # batch sum in one epoch
        sample_size += len(x_)

        # etismate epoch cost
        one_loop_time = time.time() - loop_time_start
        one_loop_time *= data_set_size / data_loader.batch_size
        loop_time_diff = one_loop_time * (1. - (sample_size / data_set_size))
        loop_time_diff = '{:.0f}h {:.0f}m {:.0f}s'.format(loop_time_diff // 3600,
                                                          loop_time_diff // 60,
                                                          loop_time_diff % 60)

        # print log
        stdout.write(
            "\r%s" %
            "Training: [{:5d}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.4f}\tRemaining Time: {:10s}".format(
                sample_size,
                data_set_size,
                100. * sample_size / data_set_size,
                loss.item(),
                running_corrects.double() / sample_size,
                loop_time_diff
            )
        )
        stdout.flush()

    t_ = time.time() - since
    epoch_loss = running_loss / data_set_size
    epoch_acc = running_corrects.double() / data_set_size

    # recorder
    model_recorder['train_acc'] += [epoch_acc]
    model_recorder['train_loss'] += [epoch_loss]
    # get learning rate
    optimizer_state = optimizer.state_dict()['param_groups'][0]

    model_recorder['train_lr'] += [optimizer_state['lr']]
    print()
    logging.info('Epoch Time Costs: {:.0f}m {:.0f}s'.format(t_ // 60, t_ % 60))
    print('Epoch Time Costs: {:.0f}m {:.0f}s'.format(t_ // 60, t_ % 60))
    print()
    logging.info('Train Set:\t| Average Loss: {:3.4f}\t| Accuracy: {:3.4f}\t| Learning Rate: {}'.format(
        epoch_loss,
        epoch_acc,
        optimizer_state['lr']
    ))
    print('Train Set:\t| Average Loss: {:3.4f}\t| Accuracy: {:3.4f}\t| Learning Rate: {}'.format(
        epoch_loss,
        epoch_acc,
        optimizer_state['lr']
    ))


def valid(model_, device_, data_loader, acc, wts):
    global epoch_no_improve, n_epoch_stop, stop_flag
    global model_recorder
    model_.eval()
    epoch_loss = 0
    correct = 0
    model_wts = wts
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device_), target.to(device_)
            output = model_(data)
            epoch_loss += criterion(output, target).item() * data.size(0)
            predict = output.max(1, keepdim=True)[1]
            correct += predict.eq(target.view_as(predict)).sum().item()

        epoch_loss /= len(data_loader.dataset)
        epoch_acc = float(correct) / len(data_loader.dataset)

        # recorder
        model_recorder['valid_acc'] += [epoch_acc]
        model_recorder['valid_loss'] += [epoch_loss]
        logging.info("Test Set:\t| Average Loss: {:.4f}\t| Accuracy: {:3.4f}\t|\n".format(epoch_loss, epoch_acc))
        print("Test Set:\t| Average Loss: {:.4f}\t| Accuracy: {:3.4f}\t|\n".format(epoch_loss, epoch_acc))

    if epoch_acc > acc:
        print('New high accuracy: {}'.format(epoch_acc))
        print()
        acc = epoch_acc
        torch.save(model_.state_dict(), model_dir + model.__class__.__name__ + '.pkl')
        model_wts = copy.deepcopy(model_.state_dict())
        epoch_no_improve = 0
    else:
        epoch_no_improve += 1
        if epoch_no_improve == n_epoch_stop:
            stop_flag = True
    return acc, model_wts, epoch_loss


def training_model(epochs_=25):
    early_stop_init()
    logging.info("{:20s} {:^15s} {:20s}".format('=' * 20, model.__class__.__name__, '=' * 20))
    print("{:20s} {:^15s} {:20s}".format('=' * 20, model.__class__.__name__, '=' * 20))
    print("{:20s} {:^15s} {:20s}".format('=' * 20, 'Start Training', '=' * 20))
    global model_recorder_dict, model_recorder
    # Create a empty recorder
    model_recorder = create_recorder_dict()

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs_):
        logging.info("{:20s} {:^15s} {:20s}".format('-' * 20, 'Epoch: {}'.format(epoch + 1), '-' * 20))
        print("{:20s} {:^15s} {:20s}".format('-' * 20, 'Epoch: {}'.format(epoch + 1), '-' * 20))
        train(model, device, train_data_loader)
        best_acc, best_model_wts, loss = valid(model, device, test_data_loader, best_acc, best_model_wts)
        if stop_flag:
            print('Epoch Not improve. Early Stop.')
            break

        # recorder
        model_recorder['epoch_list'] += [epoch]
        # reduce learning rate here, not in the train function
        exp_lr_scheduler.step()
        # exp_lr_scheduler.step(loss) # for ReduceLROnPlateau

    t_ = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(t_ // 60, t_ % 60))
    logging.info('Best Test Accuracy: {:4f}'.format(best_acc))
    print('Training complete in {:.0f}m {:.0f}s'.format(t_ // 60, t_ % 60))
    print('Best Test Accuracy: {:4f}'.format(best_acc))

    model_recorder_dict.update({model.__class__.__name__: model_recorder})
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def create_recorder_dict():
    return {'epoch_list': [], 'train_acc': [], 'train_loss': [], 'valid_acc': [], 'valid_loss': [], 'train_lr': []}


def early_stop_init():
    global epoch_no_improve, n_epoch_stop, stop_flag
    epoch_no_improve, n_epoch_stop, stop_flag = 0, early_stop_step, False
    pass


if __name__ == '__main__':
    # Config -------------------------------------------------
    # gray = 1, rgb = 3
    channels = 1

    # input image dimension
    img_dim = 48

    # image resize dimension
    data_dim = 48

    # batch size
    batch_size = 100

    # epochs
    num_epochs = 500

    # early stop step
    early_stop_step = 15

    # gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model dir
    model_dir = os.getcwd() + '/models/'

    # Create Early Stop
    epoch_no_improve, n_epoch_stop, stop_flag = 0, early_stop_step, False

    # Create Model Recorder Dictionary
    model_recorder_dict = {}
    model_recorder = create_recorder_dict()

    transform = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(data_dim + 2),
            transforms.CenterCrop(data_dim),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485), std=(0.229))
        ]),
        'valid': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.CenterCrop(data_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485), std=(0.229))
        ])
    }

    train_data_loader = DataLoader(
        CreateDatasetFromDataFrame(
            data_dir,
            transform=transform['train']
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    test_data_loader = DataLoader(
        CreateDatasetFromDataFrame(
            data_dir,
            transform=transform['valid'],
            validate=True,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    model_list = [
        # DeepFaceEmotion(len(mapping.keys())),
        FacialEmotionEmbedding(64, len(mapping.keys())),
    ]

    # Training Process ------------------------------------------
    for m in model_list:
        this_str = "{:20s} {:^15s} {:20s}".format('=' * 20, m.__class__.__name__, '=' * 20)
        logging.info(this_str)
        print(this_str)
        model = m.to(device)
        model.apply(weights_init)

        summary(model, (channels, img_dim, img_dim))

        criterion = nn.CrossEntropyLoss().to(device)

        optimizer = Ranger(model.parameters(), lr=0.01, weight_decay=0.)

        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=1e-10)

        model = training_model(num_epochs)

        # Save Model
        torch.save(model.state_dict(), model_dir + model.__class__.__name__ + '.pkl')

        del model
        torch.cuda.empty_cache()
