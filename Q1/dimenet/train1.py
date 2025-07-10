import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# 导入DimeNet模型
# 确保 dimenet.py 和此脚本在同一目录下
from dimenet import DimeNet

# --- 全局设置 ---
# 创建必要的输出文件夹
os.makedirs("fig", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("valset_record", exist_ok=True)


# ==================================================================
# Part 1: 从单个 .npz 文件加载数据的数据集类
# ==================================================================
class Au20NpzDataset(Dataset):
    """
    一个从单个 .npz 文件高效加载所有数据（能量、坐标、力）的数据集。
    .npz 文件应包含 'energies', 'coords', 'forces', 'atomic_numbers' 键。
    """

    def __init__(self, npz_path):
        super().__init__()
        print(f"Loading data from {npz_path}...")
        try:
            self.data_archive = np.load(npz_path)
            self.energies = torch.from_numpy(self.data_archive['energies']).float()
            self.coords = torch.from_numpy(self.data_archive['coords']).float()
            self.forces = torch.from_numpy(self.data_archive['forces']).float()
            # 假设所有构型的原子序数都相同
            self.atomic_numbers = torch.from_numpy(self.data_archive['atomic_numbers']).long()

            self.num_structures = self.energies.shape[0]
            print(f"Successfully loaded {self.num_structures} structures.")

        except FileNotFoundError:
            print(f"Error: Data file not found at {npz_path}")
            self.num_structures = 0
        except KeyError as e:
            print(f"Error: Missing key {e} in the .npz file.")
            self.num_structures = 0

    def len(self):
        return self.num_structures

    def get(self, idx):
        # 获取第 idx 个构型的数据
        energy = self.energies[idx]
        pos = self.coords[idx]
        force = self.forces[idx]

        # 创建一个 torch_geometric.data.Data 对象
        # z: 原子序数, pos: 原子坐标, y: 能量, force: 原子受到的力
        data = Data(z=self.atomic_numbers, pos=pos, y=energy.unsqueeze(0), force=force)
        return data


# ==================================================================
# Part 2: Main Training & Evaluation Function
# ==================================================================
def train_and_evaluate(
    DATA_FILE = "data/au20_annotated_dataset.npz",
    NUM_EPOCHS = 50,
    LEARNING_RATE = 1e-4,
    BATCH_SIZE = 16,
    CUTOFF_RADIUS = 6.0,
    FORCE_WEIGHT = 0.1,
    SAVEPATH = "models/best_model_force.pth",
    VAL_SET_SAVE_PATH = 'vaset_record/valset.npz',
    ):

    # 2. 数据集初始化
    dataset = Au20NpzDataset(npz_path=DATA_FILE)
    if not dataset.len() > 0:
        print("Dataset is empty. Exiting.")
        return

    # 3. 数据集划分
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"Dataset split: {train_size} for training, {val_size} for validation.")

    # 计算训练集的能量均值和标准差用于归一化
    train_energies = [data.y.item() for data in train_dataset]
    energy_mean = np.mean(train_energies)
    energy_std = np.std(train_energies)
    print(f"\n--- Data Stats (from training set) ---")
    print(f"Energy Mean: {energy_mean:.6f}")
    print(f"Energy Std Dev: {energy_std:.6f}")
    print("--- Training will predict normalized energies ---")

    # 保存验证集到/valset_record目录下的npz文件中

    print(f"\n正在导出包含 {len(val_dataset)} 个样本的验证集...")
    val_coords, val_energies, val_forces, val_atom_nums = [], [], [], []
    for i in range(len(val_dataset)):
        data_point = val_dataset[i]
        val_coords.append(data_point.pos.numpy())
        val_energies.append(data_point.y.numpy())
        val_forces.append(data_point.force.numpy())
        val_atom_nums.append(data_point.z.numpy())

    # 使用 np.savez 来保存多个数组到.npz文件
    np.savez(
        VAL_SET_SAVE_PATH,
        coords=np.array(val_coords),
        energies=np.array(val_energies).squeeze(),  # squeeze() 去掉多余的维度
        forces=np.array(val_forces),
        atomic_numbers=np.array(val_atom_nums)
    )
    print(f"验证集已成功保存到: '{VAL_SET_SAVE_PATH}'")



    # 4. DataLoader 创建
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. 模型和优化器初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DimeNet(hidden_channels=128, out_channels=1, num_blocks=6, num_bilinear=8,
                    num_spherical=7, num_radial=6, cutoff=CUTOFF_RADIUS).to(device)

    # 我们需要两个损失函数，都使用L1Loss (MAE)
    loss_fn_energy = nn.L1Loss()
    loss_fn_force = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True,
                                                           min_lr=1e-7)

    print(f"\n--- Starting Training on {device} with Batch Size {BATCH_SIZE} ---")
    print(f"Force loss weight (rho): {FORCE_WEIGHT}")

    # 6. 训练和验证循环
    best_val_energy_mae = float('inf')

    # 记录损失用于绘图
    history = {
        'epoch': [],
        'train_total_loss': [],
        'train_energy_loss': [],
        'train_force_loss': [],
        'val_energy_mae': [],
        'val_force_mae': []
    }

    for epoch in range(NUM_EPOCHS):
        # --- 训练阶段 ---
        model.train()
        epoch_train_total_loss = 0
        epoch_train_energy_loss = 0
        epoch_train_force_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            # 必须设置 positions.requires_grad=True 才能计算梯度（力）
            batch.pos.requires_grad_(True)
            optimizer.zero_grad()

            # 1. 预测归一化的能量
            pred_energy_norm = model(batch.z, batch.pos, batch.batch)

            # 2. 计算预测的力
            # 这是实现的关键：计算能量相对于位置的梯度
            # 首先，我们需要将归一化的能量转回真实尺度以获得物理上正确的力
            pred_energy_real = pred_energy_norm.squeeze() * energy_std + energy_mean

            grad_outputs = torch.ones_like(pred_energy_real)
            pred_forces_raw, = torch.autograd.grad(
                outputs=pred_energy_real,
                inputs=batch.pos,
                grad_outputs=grad_outputs,
                create_graph=True,  # 允许创建高阶梯度，对于训练是必须的
                retain_graph=True,  # 保留图结构，以便后续计算loss.backward()
            )
            # 别忘了力的定义是能量的负梯度
            pred_forces = -pred_forces_raw

            # 3. 计算组合损失
            # 能量损失（在归一化尺度上计算）
            true_energy_norm = (batch.y - energy_mean) / energy_std
            loss_energy = loss_fn_energy(pred_energy_norm.squeeze(), true_energy_norm)

            # 力的损失（在真实尺度上计算）
            loss_force = loss_fn_force(pred_forces, batch.force)

            # 总损失
            total_loss = loss_energy + FORCE_WEIGHT * loss_force

            # 4. 反向传播和优化
            total_loss.backward()
            optimizer.step()

            epoch_train_total_loss += total_loss.item()
            epoch_train_energy_loss += loss_energy.item()
            epoch_train_force_loss += loss_force.item()

        # 计算平均训练损失
        avg_train_total_loss = epoch_train_total_loss / len(train_loader)
        avg_train_energy_loss = epoch_train_energy_loss / len(train_loader)
        avg_train_force_loss = epoch_train_force_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval()
        total_val_energy_mae = 0
        total_val_force_mae = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                # 预测能量
                pred_energy_norm = model(batch.z, batch.pos, batch.batch)
                pred_energy_real = pred_energy_norm.squeeze() * energy_std + energy_mean

                # 计算能量的MAE (eV)
                total_val_energy_mae += torch.abs(pred_energy_real - batch.y).sum().item()

                # 验证时我们不需要计算力，因为这会消耗更多资源
                # 如果需要，可以取消下面的注释，但这会减慢验证速度
                # batch.pos.requires_grad_(True)
                # ... (重复力的计算过程) ...
                # total_val_force_mae += torch.abs(pred_forces - batch.force).sum().item()

        avg_val_energy_mae = total_val_energy_mae / len(val_dataset)
        # avg_val_force_mae = total_val_force_mae / (len(val_dataset) * 20) # 如果计算力，需要除以原子数

        # 记录历史数据
        history['epoch'].append(epoch + 1)
        history['train_total_loss'].append(avg_train_total_loss)
        history['train_energy_loss'].append(avg_train_energy_loss)
        history['train_force_loss'].append(avg_train_force_loss)
        history['val_energy_mae'].append(avg_val_energy_mae)

        print(
            f"Epoch {epoch + 1:03d}/{NUM_EPOCHS} | "
            f"Train Total Loss: {avg_train_total_loss:.6f} | "
            f"Val Energy MAE (eV): {avg_val_energy_mae:.6f}"
        )

        # 更新学习率
        scheduler.step(avg_val_energy_mae)

        # 保存最优模型（基于验证集的能量MAE）
        if avg_val_energy_mae < best_val_energy_mae:
            print(
                f"  -> New best model found! Val Energy MAE improved from {best_val_energy_mae:.6f} to {avg_val_energy_mae:.6f}. Saving model...")
            best_val_energy_mae = avg_val_energy_mae
            torch.save({
                'model_state_dict': model.state_dict(),
                'energy_mean': energy_mean,
                'energy_std': energy_std,
            }, SAVEPATH)

    # 7. 训练结束后绘图
    print("\n--- Training Finished ---")
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))


    # 绘制训练损失和验证Loss对比图
    ax[0].plot(history['epoch'], history['train_energy_loss'], label='Energy Train Loss (Norm)', color='blue')
    ax[0].plot(history['epoch'], history['val_energy_mae'], label='Energy Val MAE (eV)', color='orange')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Error Value')
    ax[0].set_title('Training vs. Validation (Energy)')
    ax[0].legend()
    ax[0].grid(True)
    ax[0].set_yscale('log')

    plt.tight_layout()
    fig_path = f"fig/loss_curve_force_{timestamp}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Loss curve plot saved to {fig_path}")
    plt.show()


# main
if __name__ == '__main__':
    # 生成模型保存路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    savepath = f"models/best_model_force_{timestamp}.pth"
    valset_save_path = f"valset_record/valset_{timestamp}.npz"
    print(f"Models will be saved to: {savepath}")

    train_and_evaluate(
    # 1. 参数设置
    DATA_FILE = "data/au20_annotated_dataset.npz",  # <--- 修改为您的 .npz 文件路径
    NUM_EPOCHS = 50,
    LEARNING_RATE = 1e-4,
    BATCH_SIZE = 16,
    CUTOFF_RADIUS = 6.0,
    FORCE_WEIGHT = 0.1,  # <--- 力的损失在总损失中的权重 (rho)
    SAVEPATH=savepath,
    VAL_SET_SAVE_PATH = valset_save_path
    )

    import predict1
    predict1.main(savepath,)
