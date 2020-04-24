import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch import nn
from tworaven_solver.torch_model.datasets.dataset_placeholder import Dataset_PH


def fit_model(model, X, Y, loss='MAE', batch_size=64, epoch=100, lr=1e-3, optimizer='adam'):
    # Create components for model training: Dataset, DataLoader & Optimizer & LOSS
    dataset = Dataset_PH(X, Y)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if 'sgd' == optimizer:
        optim = SGD(model.parameters(), lr=lr)
    elif 'adam' == optimizer:
        optim = Adam(model.parameters(), lr=lr)
    else:
        raise NotImplementedError('Optimizer {} is currently not supported'.format(optimizer))

    if 'MSE' == loss:
        loss_func = nn.MSELoss(reduction='sum')
    elif 'RMSE' == loss:
        loss_func = nn.MSELoss(reduction='mean')
    elif 'MAE' == loss:
        loss_func = nn.L1Loss(reduction='mean')
    else:
        raise NotImplementedError('Loss func {} is currently not supported'.format(optimizer))

    # Training process
    for _ in range(epoch):
        for _, (f, l) in enumerate(dl):
            f, l = f.float(), l.float().view(f.size(0), -1)
            out = model(f)
            batch_loss = loss_func(out, l)

            optim.zero_grad()
            batch_loss.backward()
            optim.step()

    t_loss = 0.
    for idx, (f, l) in enumerate(dl):
        f, l = f.float(), l.float().view(f.size(0), -1)
        out = model(f)
        c_loss = loss_func(out, l)
        if 'RMSE' == loss or 'MAE' == loss:
            c_loss *= f.size(0)

        t_loss += c_loss
        if 0 == idx:
            print('Real Value\tPredicted Value:')
            for _, (r, p) in enumerate(zip(l.data, out.data)):
                print("{}\t{}".format(r[0].item(), p[0].item()))

    print('The total {} loss for current model is {}'.format(loss, t_loss / float(dataset.__len__())))
