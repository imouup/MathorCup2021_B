import torch
import numpy as np
import os
import pygad  # <--- 导入pygad库
from tqdm import tqdm

# 导入DimeNet模型
# 确保 dimenet.py 和此脚本在同一目录下
from dimenet import DimeNet


# ==================================================================
# Part 1: 适应度函数设置 (与之前相同)
# ==================================================================
def setup_fitness_calculator(model_path, device, num_atoms):
    """
    加载模型和必要的统计数据，返回一个配置好的能量计算函数。
    """
    try:
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        model = DimeNet(hidden_channels=128, out_channels=1, num_blocks=6, num_bilinear=8,
                        num_spherical=7, num_radial=6, cutoff=6.0).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully.")

        energy_mean = checkpoint['energy_mean']
        energy_std = checkpoint['energy_std']
        print(f"Loaded stats: E_mean={energy_mean:.4f}, E_std={energy_std:.4f}")

        atomic_numbers = torch.full((num_atoms,), 79, dtype=torch.long, device=device)

        def calculate_energy(structure_coords):
            """
            计算单个Au结构的预测能量。
            """
            with torch.no_grad():
                if isinstance(structure_coords, np.ndarray):
                    pos = torch.from_numpy(structure_coords).float().to(device)
                else:
                    pos = structure_coords.float().to(device)

                from torch_geometric.data import Data
                data = Data(z=atomic_numbers, pos=pos)
                data.batch = torch.zeros(atomic_numbers.numel(), dtype=torch.long, device=device)
                pred_norm = model(data.z, data.pos, data.batch).squeeze()
                pred_real_energy = pred_norm * energy_std + energy_mean

                energy_item = pred_real_energy.item()
                if not np.isfinite(energy_item):
                    return 1e10  # 一个很大的正数作为惩罚

                return energy_item

        return calculate_energy

    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'")
        return None
    except Exception as e:
        print(f"An error occurred during setup: {e}")
        return None


# ==================================================================
# Part 2: pygad 的适配与配置 (与之前相同)
# ==================================================================

# 全局变量
best_fitness_so_far = -float('inf')
NUM_ATOMS = 32


def fitness_func_pygad(ga_instance, solution, solution_idx):
    coords = np.reshape(solution, (NUM_ATOMS, 3))
    min_dist = 2.2
    dist_matrix = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(dist_matrix, np.inf)
    violations = dist_matrix[dist_matrix < min_dist]
    penalty = np.sum((min_dist - violations) ** 2) * 100.0
    energy = energy_calculator(coords)
    LOWER_ENERGY_BOUND = -2600.0
    if energy < LOWER_ENERGY_BOUND:
        return -1e10
    penalized_energy = energy + penalty
    fitness = -penalized_energy
    return fitness


def on_generation(ga_instance):
    global best_fitness_so_far
    current_best_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    print(f"Generation {ga_instance.generations_completed:03d} | "
          f"Best Fitness (Negative Energy): {current_best_fitness:.6f}", end="")
    if current_best_fitness > best_fitness_so_far:
        best_fitness_so_far = current_best_fitness
        print("  <-- New best!")
    else:
        print("")


def create_initial_population_pygad(population_size, npz_path, num_elites):
    initial_population = []
    print("\n--- Creating Initial Population for PyGAD ---")
    try:
        data_archive = np.load(npz_path)
        energies = data_archive['energies']
        coords = data_archive['coords']
        if coords.shape[1] != NUM_ATOMS or coords.shape[2] != 3:
            raise ValueError(
                f"Coordinates in {npz_path} have shape {coords.shape}, but expected ({num_elites}, {NUM_ATOMS}, 3).")
        sorted_indices = np.argsort(energies)
        elite_indices = sorted_indices[:num_elites]
        for idx in elite_indices:
            initial_population.append(coords[idx].flatten())
        print(f"Added {len(elite_indices)} elite structures (lowest energy) from {npz_path}.")
    except Exception as e:
        print(f"Warning: Could not load elite structures from '{npz_path}'. Reason: {e}")
        print("Continuing with a fully random initial population.")
        num_elites = 0
    num_random = population_size - len(initial_population)
    print(f"Generating {num_random} random structures...")
    for _ in range(num_random):
        candidate_coords = (np.random.rand(NUM_ATOMS, 3) - 0.5) * 10.0
        initial_population.append(candidate_coords.flatten())
    return initial_population


def crossover_func(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) < offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].reshape(NUM_ATOMS, 3)
        parent2 = parents[(idx + 1) % parents.shape[0], :].reshape(NUM_ATOMS, 3)
        normal = np.random.rand(3)
        if np.linalg.norm(normal) > 0:
            normal /= np.linalg.norm(normal)
        else:
            normal = np.array([1.0, 0, 0])
        p1_side_A = np.dot(parent1, normal) > 0
        p2_side_A = np.dot(parent2, normal) > 0
        parent1_A, parent1_B = parent1[p1_side_A], parent1[~p1_side_A]
        parent2_A, parent2_B = parent2[p2_side_A], parent2[~p2_side_A]
        child1 = np.vstack((parent1_A, parent2_B))
        if len(child1) > NUM_ATOMS:
            indices_to_keep = np.random.choice(len(child1), NUM_ATOMS, replace=False)
            child1 = child1[indices_to_keep]
        elif len(child1) < NUM_ATOMS:
            needed = NUM_ATOMS - len(child1)
            donor_pool = np.vstack((parent1_B, parent2_A))
            if len(donor_pool) < needed:
                offspring.append(parent1.flatten())
            else:
                donor_indices = np.random.choice(len(donor_pool), needed, replace=False)
                child1 = np.vstack((child1, donor_pool[donor_indices]))
                offspring.append(child1.flatten())
        else:
            offspring.append(child1.flatten())
        idx += 1
    return np.array(offspring)


def mutation_func(offspring, ga_instance):
    mutation_strength = ga_instance.mutation_strength if hasattr(ga_instance, 'mutation_strength') else 0.1
    for chromosome_idx in range(offspring.shape[0]):
        if np.random.rand() < ga_instance.mutation_probability:
            random_noise = np.random.randn(NUM_ATOMS, 3) * mutation_strength
            mutated_coords = offspring[chromosome_idx, :].reshape(NUM_ATOMS, 3) + random_noise
            offspring[chromosome_idx, :] = mutated_coords.flatten()
    return offspring


# ==================================================================
# Part 3: 使用pygad的主工作流
# ==================================================================

energy_calculator = None


def run_pygad_optimization():
    global energy_calculator, best_fitness_so_far, NUM_ATOMS

    # --- 1. 参数设置 ---
    MODEL_PATH = 'models/best_model_force_20250710_123529.pth'
    NPZ_DATA_PATH = 'data/au32_initial_structures.npz'
    POPULATION_SIZE = 100
    NUM_GENERATIONS = 100
    NUM_PARENTS_MATING = 10
    NUM_ELITES_FROM_DATA = 0

    # --- 2. 准备工作 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    energy_calculator = setup_fitness_calculator(model_path=MODEL_PATH, device=DEVICE, num_atoms=NUM_ATOMS)
    if not energy_calculator:
        return

    # --- 3. 配置pygad实例 ---
    num_genes = NUM_ATOMS * 3
    initial_population = create_initial_population_pygad(POPULATION_SIZE, NPZ_DATA_PATH, NUM_ELITES_FROM_DATA)
    ga_instance = pygad.GA(
        num_generations=NUM_GENERATIONS,
        num_parents_mating=NUM_PARENTS_MATING,
        fitness_func=fitness_func_pygad,
        initial_population=initial_population,
        num_genes=num_genes,
        crossover_type=crossover_func,
        mutation_type=mutation_func,
        mutation_probability=0.4,
        parent_selection_type="tournament",
        K_tournament=3,
        on_generation=on_generation
    )
    ga_instance.mutation_strength = 0.3

    # --- 4. 运行遗传算法 ---
    print(f"\n--- Starting Genetic Algorithm for Au{NUM_ATOMS} with PyGAD ---")
    ga_instance.run()

    # --- 5. 结束与输出 ---
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    best_structure = np.reshape(solution, (NUM_ATOMS, 3))
    final_energy = energy_calculator(best_structure)

    print("\n\n" + "=" * 50)
    print("      PyGAD Optimization Finished!")
    print("=" * 50)
    print(f"Best Fitness (Negative Energy) found during GA: {solution_fitness:.6f}")
    print(f"Energy of the best structure (lowest value): {final_energy:.6f} eV")
    print("Best structure coordinates (Å):")
    print(best_structure)

    ga_instance.plot_fitness()

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    xyz_filename = f"results/best_found_Au{NUM_ATOMS}_pygad_{timestamp}.xyz"
    os.makedirs('results', exist_ok=True)
    with open(xyz_filename, "w") as f:
        f.write(f"{NUM_ATOMS}\n")
        f.write(f"Energy = {final_energy}\n")
        for atom_pos in best_structure:
            f.write(f"Au {atom_pos[0]:.8f} {atom_pos[1]:.8f} {atom_pos[2]:.8f}\n")
    print(f"\nBest structure saved to '{xyz_filename}'")

    # ==================================================================
    # NEW: Part 6: 保存排名前20%的结构到NPZ文件
    # ==================================================================
    print("\n" + "=" * 50)
    print("   Saving the top 20% of the final population...")
    print("=" * 50)

    # 获取最后一代的种群和适应度
    final_population = ga_instance.population
    final_fitness = ga_instance.last_generation_fitness

    # 计算排名前20%的数量
    num_top_individuals = int(POPULATION_SIZE * 0.2)
    if num_top_individuals == 0:
        print("Population size is too small to save top 20%. Skipping.")
        return

    # 根据适应度排序并获取前20%的索引 (适应度越高越好)
    sorted_indices = np.argsort(final_fitness)[::-1]  # [::-1] 得到降序排列
    top_indices = sorted_indices[:num_top_individuals]

    # 准备存储列表，并重新计算这些结构的纯能量
    top_structures_coords = []
    top_structures_energies = []

    print(f"Recalculating pure energies for the top {num_top_individuals} structures...")
    for index in tqdm(top_indices, desc="Processing top structures"):
        individual_solution = final_population[index]
        coords = np.reshape(individual_solution, (NUM_ATOMS, 3))

        # 计算纯能量 (无惩罚项)
        pure_energy = energy_calculator(coords)

        top_structures_coords.append(coords)
        top_structures_energies.append(pure_energy)

    # 转换为Numpy数组
    top_coords_array = np.array(top_structures_coords)
    top_energies_array = np.array(top_structures_energies)

    # 保存到 .npz 文件
    npz_filename = f"results/top_20_percent_Au{NUM_ATOMS}_{timestamp}.npz"
    np.savez(
        npz_filename,
        energies=top_energies_array,
        coords=top_coords_array
    )
    print(f"\nSuccessfully saved {len(top_structures_coords)} structures to '{npz_filename}'")


if __name__ == '__main__':
    run_pygad_optimization()