import os
import numpy as np
import ase.io
from ase.optimize import BFGS
from ase.calculators.dftb import Dftb
from tqdm import tqdm
import glob
# 新增导入：用于计算斯皮尔曼相关系数
from scipy.stats import spearmanr


def find_latest_npz_file(directory="results"):
    """在指定目录中查找最新的 top_20_percent...npz 文件。"""
    list_of_files = glob.glob(os.path.join(directory, 'top_20_percent_*.npz'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def main():
    # --- 1. 加载数据 ---
    npz_file_path = find_latest_npz_file()
    if npz_file_path is None:
        print("错误：在 'results' 文件夹中未找到 'top_20_percent_...npz' 文件。")
        print("请先运行之前的GA脚本生成该文件。")
        return

    print(f"正在从文件加载结构: {npz_file_path}")
    data = np.load(npz_file_path)
    ml_energies = data['energies']
    initial_coords = data['coords']
    num_structures = len(ml_energies)
    print(f"已成功加载 {num_structures} 个结构进行评估。")

    # --- 2. 设置 ASE/DFTB 计算器 ---
    print("\n正在设置 DFTB+ 计算器...")
    dftb_prefix = os.environ.get('DFTB_PREFIX')
    if dftb_prefix is None:
        raise RuntimeError("DFTB_PREFIX 环境变量未设置。请确保您已激活正确的 conda 环境。")

    skf_path = os.path.join(dftb_prefix, 'share/dftbplus/slater-koster/')
    print(f"使用 Slater-Koster 参数文件目录: {skf_path}")

    # --- 3. DFTB 几何优化 ---
    print("\n开始对每个结构进行 DFTB 几何优化...")
    results = []

    for i in tqdm(range(num_structures), desc="DFTB Optimizations"):
        atoms = ase.Atoms('Au' + str(len(initial_coords[i])), positions=initial_coords[i])
        calculator = Dftb(
            label=f'structure_{i}',
            atoms=atoms,
            slater_koster_directory=skf_path,
            driver_arguments='GenFormat = "xyz",',
            kpts=(1, 1, 1)
        )
        atoms.set_calculator(calculator)
        optimizer = BFGS(atoms, logfile=None, trajectory=None)
        try:
            optimizer.run(fmax=0.05)
            dftb_energy = atoms.get_potential_energy()
        except Exception as e:
            print(f"\n警告：结构 {i} 的优化失败。错误: {e}。该结构将被忽略。")
            dftb_energy = np.nan

        results.append({
            'index': i,
            'ml_energy': ml_energies[i],
            'dftb_energy': dftb_energy
        })

    valid_results = [res for res in results if not np.isnan(res['dftb_energy'])]
    if len(valid_results) != len(results):
        print(f"\n注意：有 {len(results) - len(valid_results)} 个结构优化失败，已从排名中排除。")

    if not valid_results:
        print("没有成功优化的结构，无法进行分析。")
        return

    # --- 4. 分别根据 ML 能量和 DFTB 能量进行排序 ---
    print("\n所有优化已完成，正在生成排名...")
    sorted_by_ml = sorted(valid_results, key=lambda x: x['ml_energy'])
    sorted_by_dftb = sorted(valid_results, key=lambda x: x['dftb_energy'])

    # --- 5. 新增：计算统计指标 ---
    # 为了计算，首先需要将每个结构的 index 映射到它的排名
    ml_rank_map = {item['index']: rank for rank, item in enumerate(sorted_by_ml, 1)}
    dftb_rank_map = {item['index']: rank for rank, item in enumerate(sorted_by_dftb, 1)}

    # 5.1 计算平均绝对排名误差 (MARE)
    absolute_rank_errors = []
    for res in valid_results:
        struct_index = res['index']
        error = abs(ml_rank_map[struct_index] - dftb_rank_map[struct_index])
        absolute_rank_errors.append(error)
    mare = np.mean(absolute_rank_errors)

    # 5.2 计算斯皮尔曼相关系数
    # 需要确保两列能量的顺序是对应的，直接使用 valid_results 即可
    ml_energy_list = [res['ml_energy'] for res in valid_results]
    dftb_energy_list = [res['dftb_energy'] for res in valid_results]
    spearman_corr, p_value = spearmanr(ml_energy_list, dftb_energy_list)

    # --- 6. 并排输出最终排名和统计结果 ---
    print("\n\n" + "=" * 70)
    print(" " * 22 + "能量排名对比分析")
    print("=" * 70)
    print(f"{'Rank':<5} | {'ML 预测排名':<30} | {'DFTB 基准排名':<30}")
    print("-" * 70)

    for i in range(len(valid_results)):
        rank = i + 1
        ml_item = sorted_by_ml[i]
        dftb_item = sorted_by_dftb[i]
        ml_output = f"结构 #{ml_item['index']:<3} (E={ml_item['ml_energy']:.2f} eV)"
        dftb_output = f"结构 #{dftb_item['index']:<3} (E={dftb_item['dftb_energy']:.2f} eV)"
        print(f"{rank:<5} | {ml_output:<30} | {dftb_output:<30}")

    print("-" * 70)

    # 输出新增的统计分析结果
    print("\n" + "=" * 70)
    print(" " * 24 + "量化评估指标")
    print("=" * 70)
    print(f"斯皮尔曼相关系数 (Spearman's Rho): {spearman_corr:.4f}")
    print("    - 解释: 衡量两个排名的一致性。越接近 1.0，表示 ML 预测的能量排序与 DFTB 的基准排序越吻合。")
    print(f"\n平均绝对排名误差 (MARE): {mare:.4f}")
    print("    - 解释: 结构在两个列表中的平均排名差距。越接近 0，表示两个排名越一致。")
    print("-" * 70)


if __name__ == '__main__':
    # 在运行前确保 scipy 已安装: pip install scipy
    main()