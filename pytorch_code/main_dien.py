from sklearn.metrics import roc_auc_score
from utils.utils import create_dataset, Trainer
from model.DIEN import DeepInterestEvolutionNet, auxiliary_sample
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np

def train_dien(model, EPOCH, LEARNING_RATE):
    for epoch in range(EPOCH):
        train_loss = []
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        model.train()
        for batch, (x, neg_x, y) in enumerate(train_loader):
            pred, auxiliary_loss = model(x, neg_x)
            loss = criterion(pred, y.squeeze().float().detach()) + auxiliary_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        model.eval()
        val_loss = []
        prediction = []
        y_true = []
        with torch.no_grad():
            for batch, (x, y) in enumerate(val_loader):
                pred, _ = model(x)
                loss = criterion(pred, y.squeeze().float().detach())
                val_loss.append(loss.item())
                prediction.extend(pred.tolist())
                y_true.extend(y.squeeze().tolist())
        val_auc = roc_auc_score(y_true=y_true, y_score=prediction)
        print("EPOCH %s train loss : %.5f   validation loss : %.5f   validation auc is %.5f" % (
            epoch, np.mean(train_loss), np.mean(val_loss), val_auc))
    return train_loss, val_loss, val_auc



if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Training on [{}].'.format(device))

    dataset = create_dataset('amazon-books', sample_num=1000, sequence_length=20, device=device)
    field_dims, (train_X, train_y), (valid_X, valid_y), (test_X, test_y) = dataset.train_valid_test_split()

    train_X_neg = auxiliary_sample(train_X)
    train_X_neg = torch.from_numpy(train_X_neg).long()
    train_set = Data.TensorDataset(train_X, train_X_neg, train_y)
    val_set = Data.TensorDataset(valid_X, valid_y)

    EMBEDDING_DIM = 8
    LEARNING_RATE = 1e-4
    REGULARIZATION = 1e-6
    BATCH_SIZE = 2
    EPOCH = 10
    TRIAL = 100

    train_loader = Data.DataLoader(dataset=train_set,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True)
    val_loader = Data.DataLoader(dataset=val_set,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)

    dien = DeepInterestEvolutionNet(field_dims, EMBEDDING_DIM).to(device)


    optimizer = optim.Adam(dien.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)
    criterion = nn.BCELoss()

    _ = train_dien(dien, EPOCH, LEARNING_RATE)
