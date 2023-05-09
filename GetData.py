from dgl.data import FraudYelpDataset
import torch
from sklearn.model_selection import train_test_split


class GetData:
    def __init__(self,name,train_ratio):
        self.name = name
        self.train_ratio = train_ratio
        self.graph = None
        self.train_mask, self.val_mask, self.test_mask = None,None,None
        if name == 'yelp':
            dataset = FraudYelpDataset()
            self.graph = dataset[0]

        print(self.graph)
        self.graph.ndata['label'] = self.graph.ndata['label'].long().squeeze(-1)
        self.graph.ndata['feature'] = self.graph.ndata['feature'].float()

    def split_data(self):
        labels = self.graph.ndata['label']
        index = list(range(len(labels)))
        dataset_name = self.name
        if dataset_name == 'amazon':
            index = list(range(3305, len(labels)))

        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                                train_size=self.train_ratio,
                                                                random_state=2, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                test_size=0.67,
                                                                random_state=2, shuffle=True)
        train_mask = torch.zeros([len(labels)]).bool()
        val_mask = torch.zeros([len(labels)]).bool()
        test_mask = torch.zeros([len(labels)]).bool()

        train_mask[idx_train] = 1
        val_mask[idx_valid] = 1
        test_mask[idx_test] = 1

        return self.graph, train_mask, val_mask, test_mask
