import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
import os
import csv
from datetime import datetime

# 依赖库，请确保已安装 (pip install scipy)
from scipy.stats import spearmanr

# 从您提供的官方文件中导入DimeNet模型
# 确保 dimenet.py 和此脚本在同一目录下
from dimenet import DimeNet

# --- 全局设置 ---
# 创建必要的输出文件夹
os.makedirs("predict_result", exist_ok=True)


# ==================================================================
# Part 1: 数据处理 (与新的训练脚本保持一致)
# ==================================================================
class Au20NpzDataset(Dataset):
    """
    一个从单个 .npz 文件高效加载所有数据（能量、坐标、力）的数据集。
    .npz 文件应包含 'energies', 'coords', 'forces', 'atomic_numbers' 键。
    """

    def __init__(self, npz_path):
        super().__init__()
        print(f"Loading data for prediction from {npz_path}...")
        try:
            self.data_archive = np.load(npz_path)
            self.energies = torch.from_numpy(self.data_archive['energies']).float()
            self.coords = torch.from_numpy(self.data_archive['coords']).float()
            self.forces = torch.from_numpy(self.data_archive['forces']).float()
            self.atomic_numbers = torch.from_numpy(self.data_archive['atomic_numbers']).long()

            self.num_structures = self.energies.shape[0]
            print(f"Successfully loaded {self.num_structures} structures for prediction.")

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

        # 将索引作为唯一的“代号”
        identifier = torch.tensor([idx], dtype=torch.long)

        data = Data(z=self.atomic_numbers, pos=pos, y=energy.unsqueeze(0), force=force, id=identifier)
        return data


# ==================================================================
# Part 2: 核心功能函数
# ==================================================================

def predict_from_npz(model_path: str, npz_path: str) -> list:
    """
    加载模型并对 .npz 文件中的所有数据进行批量预测。

    Returns:
        一个包含每个文件详细结果的字典列表。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"Model checkpoint '{model_path}' loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'.")
        return []

    energy_mean = checkpoint['energy_mean']
    energy_std = checkpoint['energy_std']

    model = DimeNet(hidden_channels=128, out_channels=1, num_blocks=6, num_bilinear=8,
                    num_spherical=7, num_radial=6, cutoff=6.0).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset = Au20NpzDataset(npz_path=npz_path)
    if not dataset.len() > 0: return []

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_results = []
    print("\nStarting batch prediction for energy and forces...")

    # 初始化用于计算总误差的累加器
    total_energy_mae = 0
    total_force_mae = 0

    for batch in loader:
        batch = batch.to(device)
        # 预测时也需要计算梯度，所以设置 requires_grad
        batch.pos.requires_grad_(True)

        # --- 能量预测 ---
        pred_energy_norm = model(batch.z, batch.pos, batch.batch)
        pred_energy_real = pred_energy_norm.squeeze() * energy_std + energy_mean

        # --- 力的预测 ---
        pred_energy_for_grad = pred_energy_norm.squeeze() * energy_std + energy_mean
        grad_outputs = torch.ones_like(pred_energy_for_grad)

        # 在 no_grad 上下文中使用 torch.enable_grad() 来临时启用梯度计算
        with torch.enable_grad():
            pred_forces_raw, = torch.autograd.grad(
                outputs=pred_energy_for_grad,
                inputs=batch.pos,
                grad_outputs=grad_outputs,
                create_graph=False,  # 推理时不需要创建图
                retain_graph=False,
            )
        pred_forces = -pred_forces_raw

        # 累加能量和力的误差
        total_energy_mae += torch.abs(pred_energy_real - batch.y).sum().item()
        total_force_mae += torch.abs(pred_forces - batch.force).sum().item()

        # 收集每个样本的结果用于后续排名分析
        for i in range(batch.num_graphs):
            all_results.append({
                '代号': batch.id[i].item(),
                '预测能量': pred_energy_real[i].item(),
                '真实能量': batch.y[i].item()
            })

    # 计算平均误差
    avg_energy_mae = total_energy_mae / len(dataset)
    # 每个原子有3个力分量，所以总的力分量数为 len(dataset) * num_atoms * 3
    num_atoms = dataset.atomic_numbers.shape[0]
    avg_force_mae = total_force_mae / (len(dataset) * num_atoms * 3)

    print("Prediction finished.")
    # 在结果中附上总体误差
    return all_results, avg_energy_mae, avg_force_mae


def save_and_analyze(results: list, output_csv_path: str, energy_mae: float, force_mae: float):
    """
    计算排名，保存CSV，并进行能量和力的准确度分析。
    """
    if not results or len(results) < 2:
        print("Results are insufficient for analysis and saving.")
        return

    # --- 1. 计算排名 ---
    sorted_by_pred = sorted(results, key=lambda x: x['预测能量'])
    sorted_by_true = sorted(results, key=lambda x: x['真实能量'])
    pred_rank_map = {res['代号']: i + 1 for i, res in enumerate(sorted_by_pred)}
    true_rank_map = {res['代号']: i + 1 for i, res in enumerate(sorted_by_true)}

    for res in results:
        res['预测排名'] = pred_rank_map[res['代号']]
        res['真实排名'] = true_rank_map[res['代号']]

    # --- 2. 保存包含完整信息的结果到CSV文件 ---
    results.sort(key=lambda x: x['真实排名'])
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            fieldnames = ['代号', '真实能量', '预测能量', '真实排名', '预测排名']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for res in results:
                writer.writerow({
                    '代号': res['代号'],
                    '真实能量': f"{res['真实能量']:.6f}",
                    '预测能量': f"{res['预测能量']:.6f}",
                    '真实排名': res['真实排名'],
                    '预测排名': res['预测排名']
                })
        print(f"\nDetailed prediction results with rankings saved to: '{output_csv_path}'")
    except IOError:
        print(f"\nError: Could not write results to '{output_csv_path}'")

    # --- 3. 进行排序和误差分析 ---
    rank_errors = [abs(res['预测排名'] - res['真实排名']) for res in results]
    mean_absolute_rank_error = np.mean(rank_errors)
    true_energies = [res['真实能量'] for res in results]
    pred_energies = [res['预测能量'] for res in results]
    spearman_corr, _ = spearmanr(true_energies, pred_energies)

    print("\n" + "=" * 60)
    print(" " * 22 + "Prediction Accuracy Analysis")
    print("=" * 60)
    print("--- Energy Prediction ---")
    print(f"  - Mean Absolute Error (MAE):    {energy_mae:.6f} eV")
    print("--- Force Prediction ---")
    print(f"  - Mean Absolute Error (MAE):    {force_mae:.6f} eV/Å")
    print("--- Ranking Accuracy ---")
    print(f"  - Mean Absolute Rank Error:     {mean_absolute_rank_error:.2f}")
    print(f"  - Spearman Correlation (ρ):     {spearman_corr:.6f}")
    print("=" * 60)


# ==================================================================
# Part 3: 脚本主入口
# ==================================================================
def main(model_path=''):
    """主函数，用于设置路径并调用所有流程。"""
    if not model_path:
        print("Error: Please provide a model path.")
        return

    # --- 请在这里修改您的数据文件路径 ---
    INPUT_NPZ_FILE = "data/au20_annotated_dataset.npz"  # 包含待预测数据的.npz文件
    # ---------------------------------

    # 1. 执行预测
    results_list, energy_mae, force_mae = predict_from_npz(model_path=model_path, npz_path=INPUT_NPZ_FILE)

    if results_list:
        # 2. 保存并分析结果
        # 从模型路径中提取时间戳，用于命名结果文件
        try:
            timestamp_str = model_path.split('_')[-1].split('.')[0]
        except:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_csv_path = f"predict_result/prediction_results_force_{model_path[17:-4]}.csv"
        save_and_analyze(results_list, output_csv_path, energy_mae, force_mae)


if __name__ == '__main__':
    # --- 使用示例 ---
    # 将下面的路径替换为您通过新训练脚本保存的最新模型路径
    # 例如: 'models/best_model_force_20250710_143000.pth'
    latest_model_path = 'models/best_model_force_20250710_123529.pth'
    main(model_path=latest_model_path)
