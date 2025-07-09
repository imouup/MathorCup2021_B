import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import os
import glob
import numpy as np

# 导入DimeNet模型
from Q1.dimenet.dimenet import DimeNet


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
    BATCH_SIZE = 32
    CUTOFF_RADIUS = 5.0
    NUM_WORKERS = 4

    # 2. 数据集初始化
    dataset = Au20GeoDataset(data_dir=DATA_DIR)
    if not dataset: return

    # 3. 数据集划分
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 计算训练集的均值和标准差
    train_energies = [data.y.item() for data in train_dataset]
    energy_mean = np.mean(train_energies)
    energy_std = np.std(train_energies)
    print(f"\n--- Data Stats (from training set) ---")
    print(f"Energy Mean: {energy_mean:.6f}")
    print(f"Energy Std Dev: {energy_std:.6f}")
    print("--- Training will predict normalized energies ---")
    # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---

    # 4. DataLoader 创建
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DimeNet(hidden_channels=128, out_channels=1, num_blocks=6, num_bilinear=8,
                    num_spherical=7, num_radial=6, cutoff=CUTOFF_RADIUS).to(device)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n--- Starting Training on {device} with Batch Size {BATCH_SIZE} ---")

    # 6. 训练和验证循环
    for epoch in range(NUM_EPOCHS):

        # 训练
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # 模型预测的是归一化后的能量
            pred_norm = model(batch.z, batch.pos, batch.batch)

            # 将真实能量也进行归一化，然后计算loss
            true_norm = (batch.y - energy_mean) / energy_std
            loss = loss_fn(pred_norm.squeeze(), true_norm)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # 验证
        model.eval()
        total_val_mae_ev = 0  # 我们在验证时直接计算真实尺度的MAE (eV)
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                # 模型预测归一化能量
                pred_norm = model(batch.z, batch.pos, batch.batch)

                # 将预测结果反归一化，得到真实的能量预测值 (eV)
                pred_ev = pred_norm.squeeze() * energy_std + energy_mean

                # 在真实尺度上计算绝对误差
                abs_error = torch.abs(pred_ev - batch.y)
                total_val_mae_ev += abs_error.sum().item()

        # 计算验证集的平均绝对误差(MAE)，单位是eV
        avg_val_mae_ev = total_val_mae_ev / len(val_dataset)

        print(
            f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | Train Loss (Norm): {avg_train_loss:.6f} | Val MAE (eV): {avg_val_mae_ev:.6f}")

    # ... (最终评估逻辑与验证循环类似) ...


# ==================================================================
# Part 3: Main Execution Guard
# ==================================================================
if __name__ == '__main__':
    train_and_evaluate()