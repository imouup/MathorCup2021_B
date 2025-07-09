import torch
from dimenet1 import DimeNet
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import glob
import time


# --- 数据集处理 ---

class Au20Dataset(Dataset):
    """
    优化后的数据集。
    在加载时一次性计算并缓存每个图的边和三元组。
    """

    def __init__(self, data_dir, cutoff=5.0):
        self.cutoff = cutoff
        self.filepaths = glob.glob(os.path.join(data_dir, '*.xyz*'))
        self.data_cache = []

        print("--- Pre-processing data, this may take a moment... ---")
        start_time = time.time()
        for filepath in self.filepaths:
            parsed_data = self._parse_and_build_graph(filepath)
            if parsed_data:
                self.data_cache.append(parsed_data)
        end_time = time.time()
        print(f"--- Pre-processing finished in {end_time - start_time:.2f} seconds. ---")
        print(f"Successfully loaded and pre-processed {len(self.data_cache)} structures.")

    def _get_triplets(self, j, i, num_atoms):
        """这个函数现在是数据集类的一部分"""
        idx_j, idx_i = j, i
        i_to_j = [[] for _ in range(num_atoms)]
        for edge_idx, atom_idx in enumerate(idx_j):
            i_to_j[atom_idx.item()].append(edge_idx)
        idx_kj, idx_ji = [], []
        for edge_idx, (atom_j, atom_i) in enumerate(zip(idx_j, idx_i)):
            possible_k_edges = [k_edge for k_edge in i_to_j[atom_j.item()] if j[k_edge] != atom_i]
            idx_kj.extend(possible_k_edges)
            idx_ji.extend([edge_idx] * len(possible_k_edges))
        return torch.tensor(idx_kj), torch.tensor(idx_ji)

    def _parse_and_build_graph(self, filepath):
        # 1. 解析文件
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                num_atoms = int(lines[0].strip())
                energy = float(lines[1].strip().split()[-1])
                positions = []
                for i in range(2, 2 + num_atoms):
                    parts = lines[i].strip().split()
                    positions.append([float(p) for p in parts[1:4]])
            pos = torch.tensor(positions, dtype=torch.float)
            energy = torch.tensor([energy], dtype=torch.float)
            z = torch.zeros(num_atoms, dtype=torch.long)
        except (ValueError, IndexError):
            return None  # 如果文件格式有问题则跳过

        # 2. 构建边
        dist_matrix = torch.cdist(pos, pos)
        edge_indices = (dist_matrix > 0) & (dist_matrix < self.cutoff)
        j, i = edge_indices.nonzero(as_tuple=True)  # j是源, i是目标

        # 3. 构建三元组
        idx_kj, idx_ji = self._get_triplets(j, i, num_atoms)

        return {'z': z, 'pos': pos, 'i': i, 'j': j, 'idx_kj': idx_kj, 'idx_ji': idx_ji, 'energy': energy}

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        return self.data_cache[idx]


def collate_fn(batch):
    """
    处理图数据批次的函数。对于batch_size=1，直接返回即可。
    如果未来使用更大的batch_size，这里需要更复杂的逻辑来合并图。
    """
    return batch[0]



# --- 模型训练与评估 ---

def train_and_evaluate():
    """
    完整的训练、验证和最终评估函数。
    """
    # 1. 设置超参数
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # vvv  请务必将此路径修改为您存放.xyz文件的实际文件夹路径  vvv
    DATA_DIR = "../data/au20"
    # ^^^  请务必将此路径修改为您存放.xyz文件的实际文件夹路径  ^^^
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    NUM_EPOCHS = 50  # 训练周期数，您可以根据需要调整
    LEARNING_RATE = 1e-4  # 学习率
    BATCH_SIZE = 1  # 批处理大小，由于图结构各不相同，保持为1
    CUTOFF_RADIUS = 5.0  # 定义原子间连接的截断半径（单位：埃）

    # 2. 准备数据 (在初始化时会自动进行预处理)
    print("Initializing and pre-processing dataset...")
    dataset = Au20Dataset(data_dir=DATA_DIR, cutoff=CUTOFF_RADIUS)
    if len(dataset) == 0:
        # 如果在预处理后数据集为空，则直接退出
        return

    # 划分训练集和验证集 (80% 训练, 20% 验证)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建DataLoader
    NUM_WORKERS = 10

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)

    # 3. 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DimeNet(
        num_feat=128,
        num_blocks=6,
        num_radial=6,
        num_spherical=7,
        cutoff=CUTOFF_RADIUS
    ).to(device)

    loss_fn = nn.MSELoss()  # 均方误差损失
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. 训练和验证循环
    print("\n--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()

        # --- 训练部分 ---
        model.train()
        total_train_loss = 0
        for data in train_loader:
            # 将所有预处理好的数据张量移动到指定设备
            z, pos, i, j, idx_kj, idx_ji, energy = \
                data['z'].to(device), data['pos'].to(device), data['i'].to(device), data['j'].to(device), \
                    data['idx_kj'].to(device), data['idx_ji'].to(device), data['energy'].to(device)

            optimizer.zero_grad()
            predicted_energy = model(z, pos, i, j, idx_kj, idx_ji)
            loss = loss_fn(predicted_energy, energy.squeeze())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- 验证部分 ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                z, pos, i, j, idx_kj, idx_ji, energy = \
                    data['z'].to(device), data['pos'].to(device), data['i'].to(device), data['j'].to(device), \
                        data['idx_kj'].to(device), data['idx_ji'].to(device), data['energy'].to(device)

                predicted_energy = model(z, pos, i, j, idx_kj, idx_ji)
                loss = loss_fn(predicted_energy, energy.squeeze())
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        epoch_end_time = time.time()

        # 打印当前周期的训练和验证损失以及耗时
        print(f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"Time: {epoch_end_time - epoch_start_time:.2f}s")

    print("--- Training Finished ---")

    # 5. 在所有数据上进行最终评估，以寻找全局最优结构
    print("\nPredicting energies for all structures to find the global optimum...")
    model.eval()
    all_predictions = []
    full_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    with torch.no_grad():
        for data in full_loader:
            z, pos, i, j, idx_kj, idx_ji = \
                data['z'].to(device), data['pos'].to(device), data['i'].to(device), data['j'].to(device), \
                    data['idx_kj'].to(device), data['idx_ji'].to(device)

            predicted_energy = model(z, pos, i, j, idx_kj, idx_ji)
            all_predictions.append({
                'filepath': data['filepath'],
                'true_energy': data['energy'].item(),
                'predicted_energy': predicted_energy.item()
            })

    # 找到模型预测能量最低的结构
    best_structure = min(all_predictions, key=lambda x: x['predicted_energy'])

    print("\n--- Prediction Complete ---")
    print("Found Global Optimal Structure (based on model prediction):")
    print(f"  - File Path: {best_structure['filepath']}")
    print(f"  - True Energy from file: {best_structure['true_energy']:.6f}")
    print(f"  - Predicted Energy by Model: {best_structure['predicted_energy']:.6f}")
    print("\nTo describe its shape, please visualize the file using software like VMD.")




if __name__ == '__main__':
    # 运行主函数
    train_and_evaluate()