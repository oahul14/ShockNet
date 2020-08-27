import torch
from os.path import join as osjoin
import torchvision

def predict(model, device, mseloss, l1loss, data_loader, dirs):
    model.eval()
    for X, Y, fname in data_loader:
        with torch.no_grad():
            X, Y = X.to(device), Y.to(device)
            pred_Y = model(X)
            mse = '{0:.5f}'.format(float(mseloss(pred_Y, Y)))
            mae = '{0:.5f}'.format(float(l1loss(pred_Y, Y)))
            fnamep = '_'.join([str(mse), str(mae)]) + '_' + fname[0]
            fnames = [fname[0], fname[0], fnamep]
            for i, x in enumerate([X, Y, pred_Y]):
                torchvision.utils.save_image(x[0, :, :, :], osjoin(dirs[i], fnames[i]))