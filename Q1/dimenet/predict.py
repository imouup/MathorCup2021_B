import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
import os
import glob
import time
import csv
from datetime import datetime

# 依赖库，请确保已安装 (pip install scipy)
from scipy.stats import spearmanr

# 从您提供的官方文件中导入DimeNet模型
from dimenet import DimeNet


# ==================================================================
# Part 1: 数据处理
# ==================================================================
class Au20GeoDataset(Dataset):
    """
    数据集类：加载.xyz文件，提取坐标、真实能量和作为整数ID的代号。
    """

    def __init__(self, data_dir: str):
        super().__init__()
        # glob.glob支持通配符，可以匹配.xyz, .xyz.txt等
        self.filepaths = glob.glob(os.path.join(data_dir, '*.xyz*'))
        if not self.filepaths:
            print(f"警告: 在目录 '{data_dir}' 中未找到任何.xyz文件。")
        else:
            print(f"找到 {len(self.filepaths)} 个文件待处理。")

    def len(self):
        return len(self.filepaths)

    def get(self, idx):
        filepath = self.filepaths[idx]
        try:
            # 从文件名中提取“代号”作为整数ID
            base_filename = os.path.basename(filepath)
            identifier_str = os.path.splitext(base_filename)[0]
            identifier_int = int(identifier_str)

            with open(filepath, 'r') as f:
                lines = f.readlines()

            num_atoms = int(lines[0])
            if len(lines) < 2 + num_atoms: return None  # 文件不完整

            true_energy = float(lines[1].split()[-1])
            positions = [[float(p) for p in l.split()[1:4]] for l in lines[2:2 + num_atoms]]

            pos = torch.tensor(positions, dtype=torch.float32)
            z = torch.full((num_atoms,), 79, dtype=torch.long)  # Gold (Au)
            y = torch.tensor([true_energy], dtype=torch.float32)
            id_tensor = torch.tensor([identifier_int], dtype=torch.long)

            # 返回一个标准的Data对象，包含所有需要的信息
            return Data(z=z, pos=pos, y=y, id=id_tensor)

        except (ValueError, IndexError):
            print(f"警告: 无法解析文件 {os.path.basename(filepath)}。已跳过。")
            return None


# ==================================================================
# Part 2: 核心功能函数
# ==================================================================

def predict_folder(model_path: str, input_dir: str) -> list:
    """
    加载模型并对文件夹中的所有文件进行批量预测。

    Returns:
        一个包含每个文件详细结果的字典列表。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"模型检查点 '{model_path}' 加载成功。")
    except FileNotFoundError:
        print(f"错误: 在路径 '{model_path}' 未找到模型文件。")
        return []

    energy_mean = checkpoint['energy_mean']
    energy_std = checkpoint['energy_std']

    model = DimeNet(hidden_channels=128, out_channels=1, num_blocks=6, num_bilinear=8,
                    num_spherical=7, num_radial=6, cutoff=5.0).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset = Au20GeoDataset(data_dir=input_dir)
    # 使用torch_geometric默认的DataLoader，它能正确处理包含各种属性的Data对象
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_results = []
    print("\n开始批量预测...")
    with torch.no_grad():
        for batch in loader:
            if not batch: continue

            batch = batch.to(device)
            pred_norm = model(batch.z, batch.pos, batch.batch)
            pred_ev = pred_norm.squeeze() * energy_std + energy_mean

            for i in range(batch.num_graphs):
                all_results.append({
                    '代号': batch.id[i].item(),
                    '预测能量': pred_ev[i].item(),
                    '真实能量': batch.y[i].item()
                })

    print("预测完成。")
    return all_results


def save_and_analyze(results: list, output_csv_path: str):
    """
    计算排名，保存包含所有信息的CSV，并进行排序准确度分析。
    """
    if not results or len(results) < 2:
        print("结果不足，无法进行分析和保存。")
        return

    # --- 1. 计算排名 ---
    # 按预测能量和真实能量分别排序
    sorted_by_pred = sorted(results, key=lambda x: x['预测能量'])
    sorted_by_true = sorted(results, key=lambda x: x['真实能量'])

    # 创建从“代号”到“排名”的映射 (排名从1开始，更直观)
    pred_rank_map = {res['代号']: i + 1 for i, res in enumerate(sorted_by_pred)}
    true_rank_map = {res['代号']: i + 1 for i, res in enumerate(sorted_by_true)}

    # --- 2. 将排名信息添加到结果列表中 ---
    for res in results:
        res['预测排名'] = pred_rank_map[res['代号']]
        res['真实排名'] = true_rank_map[res['代号']]

    # --- 3. 将包含完整信息的结果保存到CSV文件 ---
    # 为了方便对比，我们按真实排名进行排序后输出
    results.sort(key=lambda x: x['真实排名'])
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            # 定义包含所有信息的表头
            fieldnames = ['代号', '真实能量', '预测能量', '真实排名', '预测排名']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # 写入所有数据
            for res in results:
                writer.writerow({
                    '代号': res['代号'],
                    '真实能量': f"{res['真实能量']:.6f}",
                    '预测能量': f"{res['预测能量']:.6f}",
                    '真实排名': res['真实排名'],
                    '预测排名': res['预测排名']
                })
        print(f"\n包含排名的详细预测结果已成功保存到: '{output_csv_path}'")
    except IOError:
        print(f"\n错误: 无法将结果写入到 '{output_csv_path}'")

    # --- 4. 进行排序准确度分析 (基于已计算好的排名) ---
    rank_errors = [abs(res['预测排名'] - res['真实排名']) for res in results]
    mean_absolute_rank_error = np.mean(rank_errors)

    true_energies = [res['真实能量'] for res in results]
    pred_energies = [res['预测能量'] for res in results]
    spearman_corr, _ = spearmanr(true_energies, pred_energies)

    print("\n" + "=" * 60)
    print(" " * 21 + "预测排序准确度分析")
    print("=" * 60)
    print(f"  - 平均绝对排名误差 (MARE):  {mean_absolute_rank_error:.2f} 位")
    print(f"    (一个结构在预测列表中的排名，平均偏离其真实排名 {mean_absolute_rank_error:.2f} 位)")
    print(f"  - 斯皮尔曼相关系数 (ρ):   {spearman_corr:.6f}")
    print(f"    (衡量两个排序的相似度，+1代表完美一致，越接近1越好)")
    print("=" * 60)


# ==================================================================
# Part 3: 脚本主入口
# ==================================================================
def main(model_path = ''):
    """主函数，用于设置路径并调用所有流程。"""
    # --- 请在这里修改您的路径 ---
    MODEL_PATH = model_path
    INPUT_DIR = "data/au20"  # 包含待预测.xyz文件的文件夹

    # 1. 执行预测，获取包含所有信息的完整结果列表
    results_list = predict_folder(model_path=MODEL_PATH, input_dir=INPUT_DIR)

    if results_list:
        # 2. 需求一：输出结果到CSV文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv_path = f"predict_result/prediction_results_{MODEL_PATH[18:-4]}.csv"
        save_and_analyze(results_list, output_csv_path)



if __name__ == '__main__':
    main('models/best_model_20250709_220816.pth')