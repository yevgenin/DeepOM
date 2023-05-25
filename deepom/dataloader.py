import torch


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, labels, dataset_name=""):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.dataset_name = dataset_name

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        # Expects to get data as a pytorch tensor
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y
