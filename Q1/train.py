import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
import os
import glob
import time

# 导入DimeNet模型
from dimenet import DimeNet


# ==================================================================
# Part 1: 使用 torch_geometric 的标准方式定义数据集
# ==================================================================
class Au20GeoDataset(Dataset):
    """
    一个简洁的数据集，只负责读取文件并创建 torch_geometric.data.Data 对象。
    图的构建将在训练时即时高效完成。
    """

    def __init__(self, data_dir):
        super().__init__()
        self.filepaths = glob.glob(os.path.join(data_dir, '*.xyz*'))
        print(f"Found {len(self.filepaths)} files.")

    def len(self):
        return len(self.filepaths)

    def get(self, idx):
        filepath = self.filepaths[idx]

        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()

            if len(lines) < 22: return None

            num_atoms = int(lines[0])
            energy = float(lines[1].split()[-1])

            positions = []
            for i in range(2, 2 + num_atoms):
                parts = lines[i].strip().split()
                positions.append([float(p) for p in parts[1:4]])

            pos = torch.tensor(positions, dtype=torch.float32)
            y = torch.tensor([energy], dtype=torch.float32)
            z = torch.full((num_atoms,), 0, dtype=torch.long)  # 使用0代表金原子

            # 创建一个Data对象
            data = Data(z=z, pos=pos, y=y, filepath=filepath)
            return data
        except (ValueError, IndexError):
            return None


# ==================================================================
# Part 2: Main Training & Evaluation Function
# ==================================================================
def train_and_evaluate():
    # 1. 参数设置
    DATA_DIR = "data/au20"
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 32  # 现在可以安全地使用大batch size
    CUTOFF_RADIUS = 5.0

    # 2. 数据集初始化和加载
    # 注意：这里我们使用了 torch_geometric.loader.DataLoader
    dataset = Au20GeoDataset(data_dir=DATA_DIR)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # PyG的DataLoader会自动处理图的批处理
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 使用官方DimeNet模型
    model = DimeNet(
        hidden_channels=128,
        out_channels=1,  # 预测一个能量值
        num_blocks=6,
        num_bilinear=8,
        num_spherical=7,
        num_radial=6,
        cutoff=CUTOFF_RADIUS,
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n--- Starting Training on {device} with Batch Size {BATCH_SIZE} using official DimeNet ---")

    # 4. 训练和验证循环
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()

        # 训练
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.z, batch.pos, batch.batch)
            loss = loss_fn(pred.squeeze(), batch.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # 验证
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.z, batch.pos, batch.batch)
                loss = loss_fn(pred.squeeze(), batch.y)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Time: {time.time() - epoch_start_time:.2f}s")

    # ... (最终评估逻辑与此类似) ...


# ==================================================================
# Part 3: Main Execution Guard
# ==================================================================
if __name__ == '__main__':
    train_and_evaluate()