from utils.utils import create_dataset, Trainer
from model.NFM import NeuralFactorizationMachine
import torch
import torch.nn as nn
import torch.optim as optim


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Training on [{}].'.format(device))

    dataset = create_dataset('criteo', sample_num=1000, device=device)
    field_dims, (train_X, train_y), (valid_X, valid_y), (test_X, test_y) = dataset.train_valid_test_split()

    EMBEDDING_DIM = 8
    LEARNING_RATE = 1e-4
    REGULARIZATION = 1e-6
    BATCH_SIZE = 4096
    EPOCH = 600
    TRIAL = 100

    nfm = NeuralFactorizationMachine(field_dims, EMBEDDING_DIM).to(device)

    optimizer = optim.Adam(nfm.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)
    criterion = nn.BCELoss()

    trainer = Trainer(nfm, optimizer, criterion, BATCH_SIZE)
    trainer.train(train_X, train_y, epoch=EPOCH, trials=TRIAL, valid_X=valid_X, valid_y=valid_y)
    test_loss, test_auc = trainer.test(test_X, test_y)
    print('test_loss:  {:.5f} | test_auc:  {:.5f}'.format(test_loss, test_auc))