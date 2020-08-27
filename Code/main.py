# basic requirements
import random

# for loading data and stratification
import os
from os.path import join as osjoin
from os.path import dirname as dirname
from os.path import isdir as isdir
from PIL import Image

# for neural network
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage
import torch.optim as optim

from utilities import *
from data_loader import *
from train import train, trainT
from validate import validate, validateT
from predict import predict

from shocknet.ShockNet import ShockSENet_3, ShockSENet_4
from shocknet.TShockNet import TShockSENet_4

# make sure there are following directories:
# ./saved_models/ -> to save trained models
# ./experiments/  -> to save experiment txt files
# ./predictions/  -> to save predicted images
cur_dir = dirname(__file__)
model_dir = osjoin(cur_dir, "saved_models")
if not isdir(model_dir):
    os.mkdir(model_dir)        
text_dir = osjoin(cur_dir, "experiments")
if not isdir(text_dir):
    os.mkdir(text_dir)
pred_dir = osjoin(cur_dir, "predictions")
if not isdir(pred_dir):
    os.mkdir(pred_dir)

model_dic = {
    "tshocknet": TShockSENet_4(),
    "shocknet3": ShockSENet_3(),
    "shocknet4": ShockSENet_4()
}

def load_data(dname):
    """Load data as in dname saved in ../data/shock-datasets/dname/

    Args:
        dname (str): data folder name

    Raises:
        NameError: if the name is not found from ../data/

    Returns:
        Dataset: data parsed by Dataset class
    """
    data_dir = osjoin(cur_dir, dirname("../data/shock-datasets/"), dname)
    if not isdir(data_dir):
        print("%s not found as a directory" % data_dir)
        raise NameError
    tmp = ShockDataset(data_dir)
    print("Dataset size: ", len(tmp))
    ldr = DataLoader(tmp, batch_size=100, shuffle=True, num_workers=0)
    mean, std = None, None
    if "48x96" in dname:
        mean = [180.3010212,  204.5625, 181.78515625]
        std  = [84.69127464, 61.71853912, 89.18832674]     
    elif "64x128" in dname:
        mean = [180.15178385, 204.8688151,  181.6319987]
        std = [84.79378607, 61.2799176,  89.29453818]
    else:
        mean, std = get_mean_std(ldr)
    print("For %s: \nMean: %s\nStd: %s" % (data_dir, mean, std))

    transform_in = Compose([ToTensor(), transforms.Normalize(mean, std)])
    transform_out = Compose([ToTensor()])
    dataset = ShockDataset(data_dir, transform_in=transform_in, transform_out=transform_out)

    # Split the whole training data to training, validation and test sets
    train_size, val_size, test_size = 22000, 8000, len(dataset)-30000
    train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size])

    # Do the corresponding transforms to the three subsets
    train_set = Subset2Dataset(train_subset)
    val_set   = Subset2Dataset(val_subset)
    test_set   = Subset2Dataset(test_subset)

    return train_set, val_set, test_set

def load_model(start_epoch, model_type, mname):
    """Load model from scratch or from trained model

    Args:
        start_epoch (int): decides if load state dicts from saved models
        model_type (str): model type string: shocknet3, shocknet4 and tshocknet
        mname (str): model name saved

    Returns:
        model: loaded model
    """
    shocknet = model_dic[model_type]
    if start_epoch:
        shocknet.load_state_dict(torch.load(osjoin(model_dir, mname)))
    return shocknet

def get_start_epoch(ename):
    """Get the experimented epoch number

    Args:
        ename (str): saved experiment txt file

    Returns:
        int: start epoch number
    """
    start_epoch = 0
    try:
        start_epoch += sum(1 for line in open(osjoin(text_dir, ename)))
    except:
        print("Starting from first epoch")
    return start_epoch

def train_model(model, 
                device,
                mname, 
                ename,
                start_epoch, n_epochs,
                train_batch, test_batch,
                lr, weight_decay,
                train_loader, val_loader):
    optimiser = torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)
    mseloss = torch.nn.MSELoss()
    l1loss = torch.nn.L1Loss()
    for epoch in range(n_epochs):
        print("Epoch %s started..." % (epoch + 1 + start_epoch))

        train_mse, train_mae, train_l1stress = None, None, None
        val_mse, val_mae, val_l1stress = None, None, None
        if mname != "tshocknet":
            train_mse, train_mae = train(model, device, optimiser, mseloss, l1loss, train_loader)
            val_mse, val_mae = validate(model, device, mseloss, l1loss, val_loader)

        else:
            train_mse, train_mae, train_l1stress = trainT(model, device, optimiser, mseloss, l1loss, train_loader)
            val_mse, val_mae, val_l1stress = validateT(model, device, mseloss, l1loss, val_loader)

        training_info = [
                            str(start_epoch + epoch + 1), 
                            mname, 
                            str(train_batch), str(test_batch),
                            str(lr), str(weight_decay), 
                            str(train_mse), str(val_mse), 
                            str(train_mae), str(val_mae),
                            str(train_l1stress), str(val_l1stress)
                        ]
        save_text = ','.join(training_info) + '\n'
        text_dir = osjoin(text_dir, ename) 
        with open(text_dir, 'a') as outfile:
            outfile.write(save_text)

        torch.save(model.state_dict(), osjoin(model_dir, mname))
        print("Model %s saved" % mname)

def predict_results(model,
                    device,
                    test_loader):
    model = model.to(device)
    ic_dir = osjoin(pred_dir, "InitialCondition")
    gt_dir = osjoin(pred_dir, "ResultsPNG")
    pd_dir = osjoin(pred_dir, "Prediction")
    dirs = [ic_dir, gt_dir, pd_dir]
    for d in dirs:
        if not os.path.isdir(d):
            os.mkdir(d)

    mseloss = nn.MSELoss()
    l1loss = nn.L1Loss()
    predict(model, device, mseloss, l1loss, test_loader, dirs)

def main(model_type,
                mname,
                ename,
                dname,
                init,
                n_epochs=50,  
                batch_sizes=[10, 10],
                lr=0.02, 
                weight_decay=0):

    set_seed(42)
    device = set_device()

    train_set, val_set, test_set = load_data(dname)

    start_epoch = get_start_epoch(ename)
    model = load_model(start_epoch, model_type, mname)
    model = model.to(device)

    if init and start_epoch == 0:
        if init == "KNI":
            model.apply(kaiming_normal_init)
        if init == "KUI":
            model.apply(kaiming_uniform_init)
        print("Weights initialised with %s." % init)
    else:
        print("%s initialisation not implemented")
        raise NotImplementedError

    train_batch, test_batch = batch_sizes
    train_loader = DataLoader(train_set, batch_size=train_batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=test_batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    
    train_model(model, 
                device,
                mname, 
                ename,
                start_epoch, n_epochs,
                train_batch, test_batch,
                lr, weight_decay,
                train_loader, val_loader)

    if model_type != "tshocknet":
        predict_results(model,
                        device,
                        test_loader)


if __name__ == "__main__":
    model_dir = osjoin(cur_dir, "models")
    if not isdir(model_dir):
        os.mkdir(model_dir)        
    text_dir = osjoin(cur_dir, "experiments")
    if not isdir(text_dir):
        os.mkdir(text_dir)
    import argparse

    description = """\
                    Reproduce the result depending on the requirments.\n\
                    3 types of ShockNets, keywords: shocknet3, shocknet4 and tshocknet; \n\
                    2 types of initialisations, keywords: KUI, KNI; \n\
                    8 types of datasets, keywords: 48x96, 64x128, 128x256, 320x640 for nodal and boundary. \n\
                    Example: \n\
                        python main.py -s shocknet3 -m shocknet3_example.pth -e shocknet3_exp.txt -d nodal_48x96 -i KNI -l 0.0001 -b 14 14 -n 50 -w 0.01
                    """
                    
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--shocknet", "-s", default="shocknet4",
                        help="model architecture: shocknet3, shocknet4 and tshocknet")
    parser.add_argument("--mname", "-m", default="model.pth",
                        help="model name to be saved in .pth")
    parser.add_argument("--ename", "-e", default="experiment.txt",
                        help="experiment text file name for saving experiments")  
    parser.add_argument("--dname", "-d", default="nodal_48x96",
                        help="dataset to train with")  
    parser.add_argument("--init", "-i", default="KNI",
                        help="initialisation method: KNI (norm) and KUI (uniform)")  
    parser.add_argument("--lr", "-l", default=0.02, type=float, 
                        help="learning rate value")
    parser.add_argument("--batch", "-b", nargs=2, default=[14, 14], type=int,
                        help="training and validating batch sizes")
    parser.add_argument("--nepoch", "-n", default=50, type=int,
                        help="epoch number")
    parser.add_argument("--weightdecay", "-w", default=0, type=float, 
                        help="weight decay value")
  
    args = parser.parse_args()
    print(args.batch)
    main(
        args.shocknet,
        args.mname, args.ename, args.dname,
        args.init,
        args.nepoch, args.batch,
        args.lr, args.weightdecay
    )