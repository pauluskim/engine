from torch.utils.data import DataLoader


class IndexInterface:
    def __init__(self, model, dataset, batch_size):
        self.model = model
        self.data_loader = DataLoader(dataset, batch_size=batch_size)
        self.data_size = len(dataset)

    def indexing(self, output_path):
        raise NotImplementedError("Indexing method is a main method of this class!!")