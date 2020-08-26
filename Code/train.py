def train(model, device, optimiser, mseloss, l1loss, data_loader):
    model.train()
    train_mse, train_mae = 0, 0
    cnt = 1
    for X, Y, _ in data_loader:
        if cnt % 100 == 0:
            print("%sth Training MSE: %s, MAE: %s." % (cnt, train_mse, train_mae))
        X, Y = X.to(device), Y.to(device)
        optimiser.zero_grad()
        pred_Y = model(X)
        mse = mseloss(pred_Y, Y)
        mae = l1loss(pred_Y, Y)
        mse.backward(retain_graph=True)
        mae.backward()
        train_mse += mse*X.size(0)
        train_mae += mae*X.size(0)
        optimiser.step()  
        cnt += 1
    print("Training finished after %s batches." % cnt)
    train_mse /= len(data_loader.dataset)
    train_mae /= len(data_loader.dataset)
    return train_mse, train_mae

def trainT(model, device, optimiser, mseLoss, l1Loss, data_loader):
    model.train()
    train_mse, train_mae, train_l1stress = 0., 0., 0.
    cnt = 1
    for X, Y, _, log_max, log_min in data_loader:
        X, Y = X.to(device), Y.to(device)
        optimiser.zero_grad()
        pred_Y, pred_stress = model(X)
        pred_max, pred_min = pred_stress[:, 0], pred_stress[:, 1]
        mse = mseLoss(pred_Y, Y)
        mae = l1Loss(pred_Y, Y)
        l1stress = l1Loss(torch.flatten(pred_max), log_max) + l1Loss(torch.flatten(pred_min), log_min)
        l1stress.backward(retain_graph=True)
        losses = mse + mae
        losses.backward()

        train_mse += float(mse*X.size(0))
        train_mae += float(mae*X.size(0))
        train_l1stress += float(l1stress)

        optimiser.step() 
        cnt += 1
        if cnt % 20 == 0:
            print("%sth Training MSE: %s, MAE: %s, l1stress: %s, Total loss: %s;" % (cnt, train_mse, train_mae, train_l1stress, train_losses))
            # print("     logmax: %s, logmin: %s" % (log_max, log_min))

    print("Training finished after %s batches." % cnt)
    train_mse /= len(data_loader.dataset)
    train_mae /= len(data_loader.dataset)
    train_l1stress /= len(data_loader.dataset)
    return train_mse, train_mae, train_l1stress