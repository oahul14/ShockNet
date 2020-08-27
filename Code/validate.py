import torch

def validate(model, device, mseloss, l1loss, data_loader):
    model.eval()
    val_mse, val_mae = 0., 0.
    cnt = 1
    for X, Y, _ in data_loader:
        with torch.no_grad():
            if cnt % 100 == 0:
                print("%sth Training MSE: %s, MAE: %s." % (cnt, val_mse, val_mae))
            X, Y = X.to(device), Y.to(device)
            pred_Y = model(X)
            mse = mseloss(pred_Y, Y)
            mae = l1loss(pred_Y, Y)
            val_mse += mse*X.size(0)
            val_mae += mae*X.size(0) 
            cnt += 1
    print("Validation finished after %s batches." % cnt)
    val_mse /= len(data_loader.dataset)
    val_mae /= len(data_loader.dataset)
    return val_mse, val_mae

def validateT(model, device, mseLoss, l1Loss, data_loader):
    model.eval()
    val_mse, val_mae, val_l1stress = 0., 0., 0.
    cnt = 1
    for X, Y, _, log_max, log_min in data_loader:
        with torch.no_grad():
            X, Y = X.to(device), Y.to(device)
            pred_Y, pred_stress = model(X)
            pred_max, pred_min = pred_stress[:, 0], pred_stress[:, 1]
            mse = mseLoss(pred_Y, Y)
            mae = l1Loss(pred_Y, Y)
            l1stress = l1Loss(torch.flatten(pred_max), log_max) + l1Loss(torch.flatten(pred_min), log_min)

            val_mse += float(mse*X.size(0))
            val_mae += float(mae*X.size(0)) 
            val_l1stress += float(l1stress)
            cnt += 1
            if cnt % 100 == 0:
                print("%sth Training MSE: %s, MAE: %s, l1stress: %s, Total loss: %s" % (cnt, val_mse, val_mae, val_l1stress, val_losses))
    print("Validation finished after %s batches." % cnt)
    val_mse /= len(data_loader.dataset)
    val_mae /= len(data_loader.dataset)
    val_l1stress /= len(data_loader.dataset)
    return val_mse, val_mae, val_l1stress