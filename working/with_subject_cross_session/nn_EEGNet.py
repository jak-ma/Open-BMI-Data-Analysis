import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from braindecode.models import EEGNetv1
import numpy as np
from tqdm import tqdm
from working.utils.pre_process import preprocess_raw_data
import matplotlib.pyplot as plt
import time

def load_all_subjects(session=1, total=54):
    X_all, Y_all = [], []
    root_path = f'input/sess{session:02d}/sess{session:02d}'
    print("Loading all subjects data ...")

    for i in tqdm(range(1, total + 1), desc="Load all subjects"):
        base_path = f'_subj{i:02}_EEG_MI.mat'
        path = root_path + base_path
        x_train, y_train = preprocess_raw_data(path, 'train')
        x_test, y_test = preprocess_raw_data(path, 'test')
        X_all.append(np.concatenate([x_train, x_test]))
        Y_all.append(np.concatenate([y_train, y_test]))

    return X_all, Y_all  


class EEGDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y - 1, dtype=torch.long)  

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def run_one_subject(X_train, y_train, X_test, y_test, lr=1e-3, epochs=10, batch=32):
    n_ch = 20
    n_times = 626
    n_classes = 2

    train_dataset = EEGDataset(X_train, y_train)
    test_dataset = EEGDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch, shuffle=False, num_workers=8)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EEGNetv1(n_chans=n_ch, n_outputs=n_classes, n_times=n_times).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_acc = 0
    for epoch in range(1, epochs + 1):

        model.train()
        pbar1 = tqdm(train_loader, desc=f'Train | Epoch [{epoch}/{epochs}]')
        for X, y in pbar1:
            optimizer.zero_grad()
            output = model(X.to(device))
            loss = criterion(output, y.to(device))
            pbar1.set_postfix({'loss':loss.item()})
            loss.backward()
            optimizer.step()

        model.eval()
        pbar2 = tqdm(test_loader, desc=f'Test | Epoch [{epoch}/{epochs}]')
        correct = 0
        with torch.no_grad():
            for X, y in pbar2:
                output = model(X.to(device))
                pred = torch.argmax(output, dim=1)
                correct += (pred == y.to(device)).sum().item()

        acc = correct / len(test_dataset)
        print(f'Test | Epoch[{epoch}] Acc {acc:.3f}\n')
        best_acc = max(best_acc, acc)

    return best_acc

def main():
    start = time.time()
    total = 54
    X_all_s1, Y_all_s1 = load_all_subjects(session=1, total=total)
    X_all_s2, Y_all_s2 = load_all_subjects(session=2, total=total)
    
    subject_acc = []

    for test_id in range(total):
        X_train, y_train, X_test, y_test = X_all_s1[test_id], Y_all_s1[test_id], X_all_s2[test_id], Y_all_s2[test_id]
        print(f"\n========== Test Subject {test_id+1} / {total} ==========")
        acc = run_one_subject(X_train, y_train, X_test, y_test, lr=1e-3, epochs=10, batch=16)
        subject_acc.append(acc)
        print(f"Subject {test_id+1} Best Acc = {acc:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(subject_acc) + 1), subject_acc, marker='o')
    plt.xlabel("Subject ID")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Subject")
    plt.grid(True)
    plt.savefig("working/with_subject_cross_session/EEGNetv1.png")
    plt.show()

    print("\n========== Final Result ==========")
    for i, acc in enumerate(subject_acc):
        print(f"Subject {i+1}: {acc:.4f}")

    print(f"\nTotal Time: {time.time() - start:.2f}s")


if __name__ == '__main__':
    main()
