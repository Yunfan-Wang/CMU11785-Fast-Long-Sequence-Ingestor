from torch.utils.data import Dataset, DataLoader

class LobsterDataset(Dataset):
    def __init__(self, file_path, seq_len=10000):
        self.seq_len = seq_len
        # For baseline, I mock or load a subset to avoid GPU "starvation"
        pass

    def __getitem__(self, idx):
        # Return a window of seq_len events and the label (Mid-price move)
        return torch.randn(self.seq_len, 64), torch.tensor(1) # Mock