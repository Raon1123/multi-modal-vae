import torch

def is_multidata(dataB):
    return isinstance(dataB, list) or isinstance(dataB, tuple)


def unpack_data(dataB, device='cuda', model_type='vae', is_train=True):
    # dataB :: (Tensor, Idx) | [(Tensor, Idx)]
    """ Unpacks the data batch object in an appropriate manner to extract data """
    if is_multidata(dataB):
        if model_type == 'cvae' and is_train:
            new_dataB = []
            for datum in dataB:
                if torch.is_tensor(datum):
                    new_dataB.append(datum.to(device))
                else:
                    new_dataB.append([e.to(device) for e in datum])
            if isinstance(dataB, tuple):
                new_dataB = tuple(new_dataB)
        else:
            if torch.is_tensor(dataB[0]):
                if torch.is_tensor(dataB[1]):
                    return dataB[0].to(device)  # mnist, svhn, cubI
                elif is_multidata(dataB[1]):
                    return dataB[0].to(device), dataB[1][0].to(device)  # cubISft
                else:
                    raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB[1])))

            elif is_multidata(dataB[0]):
                return [d.to(device) for d in list(zip(*dataB))[0]]  # mnist-svhn, cubIS
            else:
                raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB[0])))
        return new_dataB

    elif torch.is_tensor(dataB):
        return dataB.to(device)
    else:
        raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB)))

def train_epoch(train_loader, model, optimizer, objective, device='cpu', K=20, 
                model_type='vae', aux_objective='entropy', classifier=None):
    model.train()
    train_loss = 0
    for data in train_loader:
        data = unpack_data(data, device=device, model_type=model_type, is_train=True)
        optimizer.zero_grad()
        loss = objective(model, data, K=K, aux_objective=aux_objective, classifier=classifier)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    return train_loss


def test_epoch(test_loader, model, objective, device='cpu', K=20):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = unpack_data(data, device=device, is_train=False)
            loss = -objective(model, data, K=K)
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    return test_loss