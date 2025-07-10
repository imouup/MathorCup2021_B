import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import os
import glob
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 导入DimeNet模型
from dimenet import DimeNet
global SAVEPATH

# 导入自定义数据集类
from train import Au20GeoDataset
# 从 train.py 导入 get_savepath 函数
from train import get_savepath

# mkdir
os.makedirs("fig", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("predict_result", exist_ok=True)


# train
def train_and_evaluate(DATA_DIR,SAVEPATH,NUM_EPOCHS,LEARNING_RATE,BATCH_SIZE,CUTOFF_RADIUS):

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True,min_lr=1e-7)

    print(f"\n--- Starting Training on {device} with Batch Size {BATCH_SIZE} ---")

    # 6. 训练和验证循环
    best_val_mae = float('inf')
    ## 记录mae
    train_loss = []
    val_loss = []


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
        # 记录训练集的MAE
        train_loss.append(avg_train_loss)

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

        # 更新学习率
        scheduler.step(avg_val_mae_ev)
        # 记录验证集的MAE
        val_loss.append(avg_val_mae_ev)

        print(
            f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | Train Loss (Norm): {avg_train_loss:.6f} | Val MAE (eV): {avg_val_mae_ev:.6f}")

        # 保存最优模型
        if avg_val_mae_ev < best_val_mae:
            best_val_mae = avg_val_mae_ev
            # 使用带时间戳的文件名进行保存
            torch.save({
                'model_state_dict': model.state_dict(),
                'energy_mean': energy_mean,
                'energy_std': energy_std,
            }, SAVEPATH)


    # 绘图
    fig, ax = plt.subplots(figsize=(8, 5))

    x_vals = list(range(1, len(train_loss) + 1))
    ax.plot(x_vals, train_loss, label='Train Loss', color='blue')
    ax.plot(x_vals, val_loss, label='Validation Loss', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True)

    fig.savefig(f"fig/loss_curve_{SAVEPATH[18:-4]}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)



# main
if __name__ == '__main__':
    savepath = get_savepath()
    train_and_evaluate(
        DATA_DIR = "data/au20",
        SAVEPATH = savepath,
        NUM_EPOCHS = 10,
        LEARNING_RATE = 1e-4,
        BATCH_SIZE = 32,
        CUTOFF_RADIUS = 6.0)
    import predict
    predict.main(savepath)
