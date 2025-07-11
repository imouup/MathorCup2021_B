import os
import ase.io
from ase.optimize import BFGS
from ase.calculators.dftb import Dftb

# --- 自动查找 Slater-Koster 参数文件的路径 ---
# 1. 从环境变量中获取 DFTB+ 的安装路径
# 这是 conda 安装 DFTB+ 带来的巨大便利
dftb_prefix = os.environ.get('DFTB_PREFIX')

if dftb_prefix is None:
    raise RuntimeError(
        "DFTB_PREFIX 环境变量未设置. "
        "请确保您已激活安装了 DFTB+ 的 conda 环境, "
        "或者手动指定您的 Slater-Koster 文件路径."
    )

# 2. 构建通用的参数文件目录路径
# conda-forge 包通常将参数文件放在这个标准位置
skf_path = os.path.join(dftb_prefix, 'share/dftbplus/slater-koster/')
print(f"自动寻找到的 Slater-Koster 参数文件目录: {skf_path}")

# --- ASE 计算流程 (与之前基本相同) ---

# 读取你的结构文件
atoms = ase.io.read('results/best_found_Au32_pygad_....xyz')

# 设置 DFTB+ 计算器 (现在路径是动态获取的)
calculator = Dftb(
    label='au32_dftb_opt',
    atoms=atoms,
    # 直接使用我们上面自动构建的路径
    # 注意：这里我们提供了整个目录，DFTB+ 会自动根据元素寻找对应的 .skf 文件
    slater_koster_directory=skf_path,
    # 如果您想指定具体的某个文件，也可以用之前的方法：
    # slater_koster_files={'Au-Au': os.path.join(skf_path, 'au-au.skf')},
    driver_arguments='GenFormat = "xyz",',
    kpts=(1, 1, 1)
)

# 将计算器“附加”到 atoms 对象上
atoms.set_calculator(calculator)

# 创建一个优化器并运行几何优化
optimizer = BFGS(atoms, trajectory='au32_opt.traj', logfile='optimizer.log')
optimizer.run(fmax=0.05)

# 优化结束后，输出结果
print('DFTB 优化完成!')
final_energy = atoms.get_potential_energy()
print(f'DFTB 优化后的能量: {final_energy:.6f} eV')

# 保存优化后的结构
ase.io.write('au32_dftb_optimized.xyz', atoms)