import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial import ConvexHull, cKDTree
from scipy.ndimage import gaussian_filter
from skimage import measure, morphology
from collections import deque
from tqdm import tqdm
import json
import warnings
import plotly.graph_objects as go

# 设置支持中文的字体（优先使用常见的中文字体）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Lucida Grande']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

warnings.filterwarnings('ignore')


# ===================== 1. 理论分析：体素空间与颗粒数量关系 =====================
class VoxelCapacityAnalyzer:
    """分析不同体素空间下可容纳的颗粒数量"""

    @staticmethod
    def analyze_capacity(voxel_sizes=[256, 512, 1024, 2048]):
        """
        分析不同体素空间的容纳能力

        Parameters:
        -----------
        voxel_sizes : list
            体素空间边长列表

        Returns:
        --------
        dict : 分析结果
        """
        results = {}

        print("=" * 80)
        print("体素空间与颗粒容纳能力分析")
        print("=" * 80)

        for size in voxel_sizes:
            total_voxels = size ** 3

            # 分析不同复杂度颗粒
            analysis = {
                'voxel_size': size,
                'total_voxels': total_voxels,
                'particle_types': {}
            }

            # 最小可分辨颗粒：5体素直径（边缘模糊占用）
            min_diameter = 5

            # 考虑边缘模糊：实际占用 = 颗粒直径 + 2*模糊区域(~10体素)
            blur_margin = 10

            particle_configs = [
                ('简单多面体(4-8面)', min_diameter, 1.0),
                ('中等多面体(9-20面)', min_diameter * 1.5, 1.2),
                ('复杂多面体(21-50面)', min_diameter * 2.0, 1.4),
                ('高复杂多面体(51-100面)', min_diameter * 2.5, 1.6),
                ('近似椭球体', min_diameter * 3.0, 1.8)
            ]

            for ptype, base_diameter, complexity_factor in particle_configs:
                # 实际占用直径
                effective_diameter = base_diameter * complexity_factor + blur_margin
                effective_radius = effective_diameter / 2

                # 单颗粒占用体积（考虑球形包络）
                particle_volume = (4 / 3) * np.pi * (effective_radius ** 3)

                # 随机堆积效率（约64%）
                packing_efficiency = 0.64

                # 最大容纳数量
                max_particles = int((total_voxels * packing_efficiency) / particle_volume)

                # 考虑孔隙率要求（0.00001 - 0.0003）
                # 固体率 = 0.9997 - 0.99999
                for target_porosity in [0.00001, 0.0001, 0.0003]:
                    solid_fraction = 1 - target_porosity
                    actual_particles = int(max_particles * solid_fraction / packing_efficiency)

                    key = f"{ptype}_porosity_{target_porosity}"
                    analysis['particle_types'][key] = {
                        'particle_type': ptype,
                        'base_diameter': base_diameter,
                        'effective_diameter': effective_diameter,
                        'complexity_factor': complexity_factor,
                        'target_porosity': target_porosity,
                        'max_particles': actual_particles,
                        'particle_volume': particle_volume
                    }

            results[size] = analysis

            print(f"\n体素空间: {size}³ = {total_voxels:,} 体素")
            print("-" * 80)
            print(f"{'颗粒类型':<25} {'孔隙率':<10} {'最大颗粒数':<15} {'有效直径':<15}")
            print("-" * 80)

            for key, data in analysis['particle_types'].items():
                if '0.0003' in key:  # 只显示目标孔隙率
                    print(f"{data['particle_type']:<25} {data['target_porosity']:<10.5f} "
                          f"{data['max_particles']:<15,} {data['effective_diameter']:<15.1f}")

        print("\n" + "=" * 80)
        print("关键结论：")
        print("1. 最小可分辨颗粒直径：5体素（考虑边缘模糊）")
        print("2. 边缘模糊区域：平均5体素，方差2体素（正态分布）")
        print("3. 颗粒复杂度与数量呈反比关系")
        print("4. 极低孔隙率下，需要大量高密度颗粒堆积")
        print("=" * 80 + "\n")

        return results


# ===================== 2. 多面体生成器（支持4-100面体） =====================
class PolyhedronGenerator:
    """生成各种复杂度的多面体"""

    @staticmethod
    def generate_polyhedron(n_faces, radius=1.0):
        """
        生成n面体

        Parameters:
        -----------
        n_faces : int
            面数（4-100）
        radius : float
            外接球半径

        Returns:
        --------
        vertices : ndarray
            顶点坐标
        """
        if n_faces == 4:  # 四面体
            vertices = np.array([
                [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
            ])
        elif n_faces == 5:  # 五面体（三角双锥）
            vertices = np.array([
                [0, 0, 1], [0, 0, -1],
                [1, 0, 0], [np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3), 0],
                [np.cos(4 * np.pi / 3), np.sin(4 * np.pi / 3), 0]
            ])
        elif n_faces == 6:  # 六面体（立方体）
            vertices = np.array([
                [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
            ])
        elif n_faces == 8:  # 八面体
            vertices = np.array([
                [1, 0, 0], [-1, 0, 0], [0, 1, 0],
                [0, -1, 0], [0, 0, 1], [0, 0, -1]
            ])
        elif n_faces == 12:  # 十二面体
            phi = (1 + np.sqrt(5)) / 2
            vertices = np.array([
                [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                [0, phi, 1 / phi], [0, phi, -1 / phi], [0, -phi, 1 / phi], [0, -phi, -1 / phi],
                [1 / phi, 0, phi], [1 / phi, 0, -phi], [-1 / phi, 0, phi], [-1 / phi, 0, -phi],
                [phi, 1 / phi, 0], [phi, -1 / phi, 0], [-phi, 1 / phi, 0], [-phi, -1 / phi, 0]
            ])
        elif n_faces == 20:  # 二十面体
            phi = (1 + np.sqrt(5)) / 2
            vertices = np.array([
                [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
                [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
                [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
            ])
        else:
            # 对于其他面数，使用球面均匀分布近似
            n_points = max(n_faces // 2, 12)

            # Fibonacci球面采样
            indices = np.arange(0, n_points, dtype=float) + 0.5
            phi = np.arccos(1 - 2 * indices / n_points)
            theta = np.pi * (1 + 5 ** 0.5) * indices

            vertices = np.column_stack([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])

        # 归一化到指定半径
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True) * radius

        return vertices


# ===================== 3. 模糊边缘法核心算法 =====================
class BlurredEdgeMethod:
    """模糊边缘法生成孔隙结构"""

    def __init__(self, vol_size=256, target_porosity=0.0003,
                 particle_diameter=10, compactness=0.95,
                 blur_mean=5, blur_std=2, save_dir='output_blurred'):
        """
        初始化

        Parameters:
        -----------
        vol_size : int
            体素空间边长
        target_porosity : float
            目标孔隙率（0.00001 - 0.0003）
        particle_diameter : int
            颗粒等效直径（体素）
        compactness : float
            密实度（0-1）
        blur_mean : float
            边缘模糊平均值（体素）
        blur_std : float
            边缘模糊标准差（体素）
        """
        self.vol_size = vol_size
        self.target_porosity = target_porosity
        self.particle_diameter = particle_diameter
        self.compactness = compactness
        self.blur_mean = blur_mean
        self.blur_std = blur_std
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        # 计算所需颗粒数
        self.n_particles = self._calculate_particle_number()

        self.particles = []
        self.volume = None
        self.gap_mask = None
        self.gap_regions = None
        self.widest_path = None

    def _calculate_particle_number(self):
        """计算达到目标孔隙率所需的颗粒数"""
        total_voxels = self.vol_size ** 3
        target_solid_voxels = total_voxels * (1 - self.target_porosity)

        # 单颗粒体积（考虑密实度）
        radius = (self.particle_diameter / 2) * self.compactness
        particle_volume = (4 / 3) * np.pi * (radius ** 3)

        n_particles_theory = int(target_solid_voxels / particle_volume)

        # 考虑空间限制：计算实际可容纳的最大颗粒数
        # 最小间距（包括边缘模糊）
        min_separation = self.particle_diameter + self.blur_mean * 2

        # 估算可容纳的颗粒数（简单立方堆积）
        particles_per_dimension = int(self.vol_size / min_separation)
        max_particles = particles_per_dimension ** 3

        # 取理论值和实际空间限制的较小值
        n_particles = min(n_particles_theory, max_particles)

        print(f"目标孔隙率: {self.target_porosity:.5f}")
        print(f"理论所需颗粒数: {n_particles_theory}")
        print(f"空间限制最大颗粒数: {max_particles}")
        print(f"实际使用颗粒数: {n_particles}")

        # 如果差距过大，给出警告
        if n_particles < n_particles_theory * 0.5:
            print(f"⚠️  警告：空间限制导致颗粒数不足，可能无法达到目标孔隙率")
            print(f"💡 建议：")
            print(f"   1. 增大体素空间尺寸（当前：{self.vol_size}）")
            print(f"   2. 减小颗粒直径（当前：{self.particle_diameter}）")
            print(f"   3. 减小边缘模糊参数（当前：{self.blur_mean}±{self.blur_std}）")

        return n_particles

    def generate_particles(self):
        """生成颗粒（不共享面和边）"""
        print(f"生成 {self.n_particles} 个颗粒...")

        # 动态调整最小间距策略
        # 对于极低孔隙率，需要颗粒更紧密排列
        if self.target_porosity < 0.001:
            # 极低孔隙率：颗粒需要紧密排列，减小间距要求
            min_separation = self.particle_diameter * 0.8 + self.blur_mean
            print(f"极低孔隙率模式：使用较小间距 {min_separation:.2f} 体素")
        else:
            min_separation = self.particle_diameter * 1.2 + self.blur_mean * 2

        centers = []
        kdtree = None

        # 增加尝试次数
        max_attempts = max(self.n_particles * 500, 50000)
        attempts = 0

        # 随机生成面数（4-100面体）
        face_distribution = np.random.choice(
            list(range(4, 101)),
            size=self.n_particles * 2,  # 生成更多备用
            p=self._get_face_probability_distribution()
        )

        pbar = tqdm(total=self.n_particles, desc="生成颗粒中心")

        # 优化策略：先在网格点附近生成，确保覆盖
        grid_spacing = min_separation * 1.1
        grid_size = int(self.vol_size / grid_spacing)

        # 第一阶段：网格化生成（确保基本覆盖）
        margin = self.particle_diameter
        if grid_size >= 2:
            grid_points = []
            for i in range(grid_size):
                for j in range(grid_size):
                    for k in range(grid_size):
                        x = margin + i * grid_spacing + np.random.uniform(-grid_spacing * 0.2, grid_spacing * 0.2)
                        y = margin + j * grid_spacing + np.random.uniform(-grid_spacing * 0.2, grid_spacing * 0.2)
                        z = margin + k * grid_spacing + np.random.uniform(-grid_spacing * 0.2, grid_spacing * 0.2)

                        # 确保在边界内
                        x = np.clip(x, margin, self.vol_size - margin)
                        y = np.clip(y, margin, self.vol_size - margin)
                        z = np.clip(z, margin, self.vol_size - margin)

                        grid_points.append(np.array([x, y, z]))

            # 随机打乱
            np.random.shuffle(grid_points)

            # 使用网格点
            for center in grid_points:
                if len(centers) >= self.n_particles:
                    break

                if kdtree is None:
                    centers.append(center)
                    kdtree = cKDTree(centers)
                    pbar.update(1)
                else:
                    dist, _ = kdtree.query(center)
                    if dist > min_separation * 0.9:  # 稍微放松要求
                        centers.append(center)
                        kdtree = cKDTree(centers)
                        pbar.update(1)

        # 第二阶段：随机填充剩余空间
        while len(centers) < self.n_particles and attempts < max_attempts:
            # 随机生成中心
            margin = self.particle_diameter
            center = np.array([
                np.random.uniform(margin, self.vol_size - margin),
                np.random.uniform(margin, self.vol_size - margin),
                np.random.uniform(margin, self.vol_size - margin)
            ])

            # 检查与已有颗粒的距离
            if kdtree is None:
                centers.append(center)
                kdtree = cKDTree(centers)
                pbar.update(1)
            else:
                dist, _ = kdtree.query(center)
                # 动态调整接受标准
                accept_threshold = min_separation * (1.0 - 0.3 * len(centers) / self.n_particles)
                if dist > accept_threshold:
                    centers.append(center)
                    kdtree = cKDTree(centers)
                    pbar.update(1)

            attempts += 1

        pbar.close()

        actual_generated = len(centers)
        if actual_generated < self.n_particles:
            print(f"⚠️  警告: 仅生成了 {actual_generated} 个颗粒（目标{self.n_particles}）")
            print(f"生成率: {100 * actual_generated / self.n_particles:.1f}%")
            self.n_particles = actual_generated
        else:
            print(f"✓ 成功生成 {actual_generated} 个颗粒")

        # 生成颗粒详细信息
        for i, center in enumerate(centers):
            n_faces = face_distribution[i % len(face_distribution)]
            radius = (self.particle_diameter / 2) * np.random.uniform(0.9, 1.1)

            # 生成多面体
            vertices = PolyhedronGenerator.generate_polyhedron(n_faces, radius)

            # 随机旋转
            angles = np.random.uniform(0, 2 * np.pi, 3)
            Rx = self._rotation_matrix_x(angles[0])
            Ry = self._rotation_matrix_y(angles[1])
            Rz = self._rotation_matrix_z(angles[2])
            R = Rz @ Ry @ Rx

            vertices = vertices @ R.T + center

            self.particles.append({
                'center': center,
                'vertices': vertices,
                'n_faces': n_faces,
                'radius': radius
            })

        # 统计颗粒间距
        if len(centers) > 1:
            distances = []
            sample_size = min(100, len(centers))
            sample_indices = np.random.choice(len(centers), sample_size, replace=False)

            for i in sample_indices:
                dist, _ = kdtree.query(centers[i], k=2)
                distances.append(dist[1])  # 最近邻距离

            print(f"颗粒间距统计: 最小={np.min(distances):.2f}, "
                  f"平均={np.mean(distances):.2f}, 最大={np.max(distances):.2f} 体素")

    def _get_face_probability_distribution(self):
        """获取面数的概率分布（地质统计学相似性）"""
        # 对数正态分布：更多简单多面体，少量复杂多面体
        faces = np.arange(4, 101)
        mu = np.log(12)  # 中值约为12面
        sigma = 0.8

        prob = np.exp(-(np.log(faces) - mu) ** 2 / (2 * sigma ** 2))
        prob = prob / prob.sum()

        return prob

    def _rotation_matrix_x(self, theta):
        """绕X轴旋转矩阵"""
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])

    def _rotation_matrix_y(self, theta):
        """绕Y轴旋转矩阵"""
        return np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

    def _rotation_matrix_z(self, theta):
        """绕Z轴旋转矩阵"""
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

    def create_volume_with_blurred_edges(self):
        """使用正态分布模糊边缘创建体积"""
        print("创建三维体积（模糊边缘）...")

        # 初始化为空（孔隙）
        self.volume = np.zeros((self.vol_size, self.vol_size, self.vol_size), dtype=np.float32)

        # 对于极低孔隙率，采用"反向思维"：先填满，再挖孔
        if self.target_porosity < 0.01:
            print("极低孔隙率模式：使用致密填充策略")
            # 先全部填充为固体
            self.volume[:] = 1.0

            # 在颗粒边界处创建细微缝隙
            gap_probability = self.target_porosity * 10  # 放大概率以补偿

            for idx, particle in enumerate(tqdm(self.particles, desc="创建颗粒间缝隙")):
                try:
                    vertices = particle['vertices']
                    hull = ConvexHull(vertices)

                    # 计算边界框（扩大以包含缝隙区域）
                    min_bounds = np.floor(vertices.min(axis=0) - self.blur_mean).astype(int)
                    max_bounds = np.ceil(vertices.max(axis=0) + self.blur_mean).astype(int)

                    min_bounds = np.clip(min_bounds, 0, self.vol_size - 1)
                    max_bounds = np.clip(max_bounds, 0, self.vol_size)

                    # 只在边界壳层创建缝隙
                    z_range = range(min_bounds[2], max_bounds[2])
                    y_range = range(min_bounds[1], max_bounds[1])
                    x_range = range(min_bounds[0], max_bounds[0])

                    if len(z_range) == 0 or len(y_range) == 0 or len(x_range) == 0:
                        continue

                    zz, yy, xx = np.meshgrid(z_range, y_range, x_range, indexing='ij')
                    local_grid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

                    # 计算到颗粒表面的距离
                    equations = hull.equations
                    inside_values = local_grid @ equations[:, :3].T + equations[:, 3]
                    inside = np.all(inside_values <= 1e-10, axis=1)

                    # 找到表面附近的点（边界层）
                    min_dist_to_surface = np.abs(inside_values).min(axis=1)

                    for i, (x, y, z) in enumerate(local_grid):
                        # 只在颗粒表面附近（距离 < blur_mean）创建缝隙
                        dist = min_dist_to_surface[i]

                        if dist < self.blur_mean:
                            # 使用正态分布控制缝隙生成概率
                            gap_prob = np.exp(-(dist - self.blur_mean) ** 2 / (2 * self.blur_std ** 2))

                            # 随机决定是否创建缝隙
                            if np.random.random() < gap_prob * gap_probability:
                                self.volume[z, y, x] = 0  # 创建缝隙

                except Exception as e:
                    continue

        else:
            # 常规孔隙率：原有算法
            for idx, particle in enumerate(tqdm(self.particles, desc="填充颗粒")):
                try:
                    vertices = particle['vertices']
                    hull = ConvexHull(vertices)

                    # 计算边界框
                    min_bounds = np.floor(vertices.min(axis=0)).astype(int)
                    max_bounds = np.ceil(vertices.max(axis=0)).astype(int)

                    min_bounds = np.clip(min_bounds - 10, 0, self.vol_size - 1)
                    max_bounds = np.clip(max_bounds + 10, 0, self.vol_size)

                    # 在边界框内检查点
                    z_range = range(min_bounds[2], max_bounds[2])
                    y_range = range(min_bounds[1], max_bounds[1])
                    x_range = range(min_bounds[0], max_bounds[0])

                    if len(z_range) == 0 or len(y_range) == 0 or len(x_range) == 0:
                        continue

                    zz, yy, xx = np.meshgrid(z_range, y_range, x_range, indexing='ij')
                    local_grid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

                    # 判断点是否在凸包内
                    equations = hull.equations
                    inside = np.all(local_grid @ equations[:, :3].T + equations[:, 3] <= 1e-10, axis=1)

                    # 计算到表面的距离（用于模糊边缘）
                    for i, (x, y, z) in enumerate(local_grid):
                        if inside[i]:
                            # 计算到最近表面的距离
                            distances = np.abs(vertices @ equations[:, :3].T + equations[:, 3])
                            min_dist = distances.min()

                            # 使用正态分布控制边缘模糊
                            blur_value = np.exp(-(min_dist - self.blur_mean) ** 2 / (2 * self.blur_std ** 2))

                            # 叠加到体积
                            self.volume[z, y, x] = max(self.volume[z, y, x], 1.0 - blur_value)

                except Exception as e:
                    continue

            # 应用整体高斯模糊（模拟边缘模糊效果）
            self.volume = gaussian_filter(self.volume, sigma=self.blur_std / 2)

        # 二值化：阈值设定
        threshold = 0.5
        self.gap_mask = (self.volume < threshold).astype(np.uint8)

        actual_porosity = self.gap_mask.sum() / self.gap_mask.size
        print(f"\n实际孔隙率: {actual_porosity:.6f}")
        print(f"目标孔隙率: {self.target_porosity:.6f}")
        print(f"偏差: {abs(actual_porosity - self.target_porosity):.6f}")

        # 如果偏差太大，尝试调整阈值
        if abs(actual_porosity - self.target_porosity) > self.target_porosity * 2:
            print("\n尝试通过调整阈值优化孔隙率...")

            # 二分搜索最佳阈值
            best_threshold = threshold
            best_diff = abs(actual_porosity - self.target_porosity)

            for t in np.linspace(0.1, 0.9, 20):
                test_mask = (self.volume < t).astype(np.uint8)
                test_porosity = test_mask.sum() / test_mask.size
                diff = abs(test_porosity - self.target_porosity)

                if diff < best_diff:
                    best_diff = diff
                    best_threshold = t

            if best_threshold != threshold:
                print(f"找到更优阈值: {best_threshold:.3f}")
                self.gap_mask = (self.volume < best_threshold).astype(np.uint8)
                actual_porosity = self.gap_mask.sum() / self.gap_mask.size
                print(f"优化后孔隙率: {actual_porosity:.6f}")
                print(f"新偏差: {abs(actual_porosity - self.target_porosity):.6f}")

    def extract_gap_regions(self):
        """提取缝隙区域"""
        print("提取缝隙区域...")
        self.gap_regions, n_regions = measure.label(
            self.gap_mask, connectivity=3, return_num=True
        )
        print(f"独立缝隙区域数: {n_regions}")

        return n_regions

    def find_widest_path(self):
        """寻找最宽缝隙路径"""
        print("计算缝隙宽度...")
        from scipy.ndimage import distance_transform_edt

        distance = distance_transform_edt(self.gap_mask)
        gap_width = distance * 2

        print("寻找最宽路径...")
        # 修复：使用 skeletonize 而不是 skeletonize_3d
        skeleton = morphology.skeletonize(self.gap_mask)
        skeleton_coords = np.argwhere(skeleton == 1)

        if len(skeleton_coords) == 0:
            print("⚠️  警告: 无有效路径（骨架为空）")
            self.widest_path = []
            return None, 0

        start = np.unravel_index(np.argmax(gap_width), gap_width.shape)

        visited = np.zeros_like(self.gap_mask, dtype=bool)
        queue = deque([(start[0], start[1], start[2], [start], [gap_width[start]])])
        max_avg = 0
        best_path = []

        max_iterations = 5000
        iteration = 0

        while queue and iteration < max_iterations:
            z, y, x, path, widths = queue.popleft()
            iteration += 1

            if visited[z, y, x]:
                continue
            visited[z, y, x] = True

            avg = np.mean(widths)
            if avg > max_avg and len(path) > 3:
                max_avg = avg
                best_path = path.copy()

            for dz, dy, dx in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
                nz, ny, nx = z + dz, y + dy, x + dx
                if (0 <= nz < self.vol_size and 0 <= ny < self.vol_size and
                        0 <= nx < self.vol_size and skeleton[nz, ny, nx] == 1 and
                        not visited[nz, ny, nx]):
                    queue.append((nz, ny, nx, path + [(nz, ny, nx)],
                                  widths + [gap_width[nz, ny, nx]]))

        self.widest_path = best_path

        if len(best_path) > 0:
            print(f"✓ 最宽路径平均宽度: {max_avg:.2f} 体素")
            print(f"  路径长度: {len(best_path)} 个点")
        else:
            print("⚠️  警告: 未找到有效路径")

        return best_path, max_avg

    def visualize_results(self):
        """可视化结果"""
        print("生成可视化...")

        # 1. 颗粒三维分布（Plotly）
        self._plot_particles_3d()

        # 2. 缝隙切片可视化
        self._plot_gap_slices()

        # 3. 最宽路径可视化
        self._plot_widest_path()

        # 4. 统计分析
        self._plot_statistics()

    def _plot_particles_3d(self):
        """绘制颗粒三维分布（改进版：多种可视化方式）"""
        print("生成颗粒三维可视化...")

        # 方法1：Plotly交互式3D散点图 + 凸包表面
        fig = go.Figure()

        # 采样显示（避免过多）
        sample_size = min(100, len(self.particles))
        sampled_indices = np.random.choice(len(self.particles), sample_size, replace=False)

        # 收集所有颗粒中心用于散点图
        centers = np.array([p['center'] for p in self.particles])
        radii = np.array([p['radius'] for p in self.particles])
        face_counts = np.array([p['n_faces'] for p in self.particles])

        # 1. 绘制颗粒中心（按面数着色）
        fig.add_trace(go.Scatter3d(
            x=centers[:, 0],
            y=centers[:, 1],
            z=centers[:, 2],
            mode='markers',
            marker=dict(
                size=radii / 2,  # 按半径缩放
                color=face_counts,
                colorscale='Viridis',
                colorbar=dict(title="面数"),
                opacity=0.8,
                line=dict(color='white', width=0.5)
            ),
            name='颗粒中心',
            text=[f'面数:{fc}, 半径:{r:.2f}' for fc, r in zip(face_counts, radii)],
            hoverinfo='text'
        ))

        # 2. 绘制部分颗粒的表面（半透明）
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']

        for i, idx in enumerate(sampled_indices[:20]):  # 只显示20个详细表面
            particle = self.particles[idx]
            vertices = particle['vertices']

            try:
                hull = ConvexHull(vertices)

                # 提取表面三角形
                x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
                i_faces, j_faces, k_faces = hull.simplices.T

                fig.add_trace(go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i_faces, j=j_faces, k=k_faces,
                    opacity=0.3,
                    color=colors[i % len(colors)],
                    flatshading=False,
                    showlegend=False,
                    hoverinfo='skip'
                ))
            except:
                continue

        fig.update_layout(
            title=dict(
                text=f"颗粒三维分布 (总数={len(self.particles)}, 显示={sample_size})<br>"
                     f"孔隙率={self.gap_mask.sum() / self.gap_mask.size:.6f}",
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis_title='X (体素)',
                yaxis_title='Y (体素)',
                zaxis_title='Z (体素)',
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=900,
            showlegend=True
        )

        fig.write_html(os.path.join(self.save_dir, "particles_3d_interactive.html"))
        print("✓ 已保存: particles_3d_interactive.html (交互式3D)")

        # 方法2：切片投影图（显示颗粒密度分布）
        self._plot_particle_density_slices()

        # 方法3：统计分析可视化
        self._plot_particle_spatial_distribution()

    def _plot_particle_density_slices(self):
        """绘制颗粒密度切片图"""
        print("生成颗粒密度切片...")

        # 创建密度场
        density = np.zeros((self.vol_size, self.vol_size, self.vol_size), dtype=np.float32)

        for particle in self.particles:
            center = particle['center'].astype(int)
            radius = int(particle['radius'])

            # 在颗粒中心周围标记
            z_min, z_max = max(0, center[2] - radius), min(self.vol_size, center[2] + radius)
            y_min, y_max = max(0, center[1] - radius), min(self.vol_size, center[1] + radius)
            x_min, x_max = max(0, center[0] - radius), min(self.vol_size, center[0] + radius)

            density[z_min:z_max, y_min:y_max, x_min:x_max] += 1

        # 绘制切片
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        slices_z = [self.vol_size // 6, self.vol_size // 3, self.vol_size // 2,
                    2 * self.vol_size // 3, 5 * self.vol_size // 6, self.vol_size - 1]

        for ax, z in zip(axes.flat, slices_z):
            im = ax.imshow(density[z, :, :], cmap='hot', origin='lower', interpolation='bilinear')
            ax.set_title(f'Z切片 = {z} (颗粒密度)', fontsize=12, fontweight='bold')
            ax.set_xlabel('X (体素)')
            ax.set_ylabel('Y (体素)')

            # 叠加颗粒中心点
            particles_in_slice = [p for p in self.particles
                                  if abs(p['center'][2] - z) < 5]
            if particles_in_slice:
                x_centers = [p['center'][0] for p in particles_in_slice]
                y_centers = [p['center'][1] for p in particles_in_slice]
                ax.scatter(x_centers, y_centers, c='cyan', s=20, marker='x', alpha=0.8)

            plt.colorbar(im, ax=ax, label='密度')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "particle_density_slices.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 已保存: particle_density_slices.png")

    def _plot_particle_spatial_distribution(self):
        """绘制颗粒空间分布统计"""
        print("生成空间分布统计图...")

        centers = np.array([p['center'] for p in self.particles])

        fig = plt.figure(figsize=(16, 12))

        # 1. XY平面投影
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(centers[:, 0], centers[:, 1], alpha=0.5, s=10, c='blue')
        ax1.set_xlabel('X (体素)')
        ax1.set_ylabel('Y (体素)')
        ax1.set_title('XY平面投影')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # 2. XZ平面投影
        ax2 = plt.subplot(2, 3, 2)
        ax2.scatter(centers[:, 0], centers[:, 2], alpha=0.5, s=10, c='green')
        ax2.set_xlabel('X (体素)')
        ax2.set_ylabel('Z (体素)')
        ax2.set_title('XZ平面投影')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

        # 3. YZ平面投影
        ax3 = plt.subplot(2, 3, 3)
        ax3.scatter(centers[:, 1], centers[:, 2], alpha=0.5, s=10, c='red')
        ax3.set_xlabel('Y (体素)')
        ax3.set_ylabel('Z (体素)')
        ax3.set_title('YZ平面投影')
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')

        # 4. X方向密度分布
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(centers[:, 0], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax4.set_xlabel('X (体素)')
        ax4.set_ylabel('颗粒数量')
        ax4.set_title('X方向分布')
        ax4.grid(True, alpha=0.3)

        # 5. Y方向密度分布
        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(centers[:, 1], bins=30, color='coral', edgecolor='black', alpha=0.7)
        ax5.set_xlabel('Y (体素)')
        ax5.set_ylabel('颗粒数量')
        ax5.set_title('Y方向分布')
        ax5.grid(True, alpha=0.3)

        # 6. Z方向密度分布
        ax6 = plt.subplot(2, 3, 6)
        ax6.hist(centers[:, 2], bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
        ax6.set_xlabel('Z (体素)')
        ax6.set_ylabel('颗粒数量')
        ax6.set_title('Z方向分布')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "particle_spatial_distribution.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 已保存: particle_spatial_distribution.png")

    def _plot_gap_slices(self):
        """绘制缝隙切片"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        slices_z = [self.vol_size // 6, self.vol_size // 3, self.vol_size // 2,
                    2 * self.vol_size // 3, 5 * self.vol_size // 6, self.vol_size - 1]

        for ax, z in zip(axes.flat, slices_z):
            im = ax.imshow(self.gap_regions[z, :, :], cmap='tab20', origin='lower')
            ax.set_title(f'Z切片 = {z} (缝隙区域)', fontsize=12, fontweight='bold')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, label='区域ID')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "gap_slices.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 已保存: gap_slices.png")

    def _plot_widest_path(self):
        """绘制最宽路径"""
        if not self.widest_path or len(self.widest_path) == 0:
            return

        fig = go.Figure()

        # 显示缝隙空间（采样）
        z, y, x = np.where(self.gap_mask > 0)
        sample_indices = np.random.choice(len(x), min(3000, len(x)), replace=False)

        fig.add_trace(go.Scatter3d(
            x=x[sample_indices], y=y[sample_indices], z=z[sample_indices],
            mode='markers',
            marker=dict(size=1, color='lightgray', opacity=0.2),
            name='缝隙空间'
        ))

        # 显示最宽路径
        path_array = np.array(self.widest_path)
        fig.add_trace(go.Scatter3d(
            x=path_array[:, 2], y=path_array[:, 1], z=path_array[:, 0],
            mode='lines+markers',
            line=dict(color='red', width=6),
            marker=dict(size=4, color='red'),
            name='最宽缝隙路径'
        ))

        fig.update_layout(
            title="三维最宽缝隙路径",
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                aspectmode='cube'
            ),
            width=900, height=900
        )

        fig.write_html(os.path.join(self.save_dir, "widest_path_3d.html"))
        print("✓ 已保存: widest_path_3d.html")

    def _plot_statistics(self):
        """绘制统计图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 颗粒面数分布
        face_counts = [p['n_faces'] for p in self.particles]
        axes[0, 0].hist(face_counts, bins=30, color='steelblue', edgecolor='black')
        axes[0, 0].set_xlabel('多面体面数')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].set_title('颗粒复杂度分布')
        axes[0, 0].grid(True, alpha=0.3)

        # 颗粒半径分布
        radii = [p['radius'] for p in self.particles]
        axes[0, 1].hist(radii, bins=30, color='coral', edgecolor='black')
        axes[0, 1].set_xlabel('颗粒半径 (体素)')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].set_title('颗粒大小分布')
        axes[0, 1].grid(True, alpha=0.3)

        # 缝隙区域大小分布
        region_sizes = [np.sum(self.gap_regions == i)
                        for i in range(1, self.gap_regions.max() + 1)]
        axes[1, 0].hist(region_sizes, bins=30, color='mediumseagreen', edgecolor='black')
        axes[1, 0].set_xlabel('缝隙区域体积 (体素)')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title('缝隙区域大小分布')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        # 孔隙率对比
        actual_p = self.gap_mask.sum() / self.gap_mask.size
        axes[1, 1].bar(['目标孔隙率', '实际孔隙率'],
                       [self.target_porosity, actual_p],
                       color=['orange', 'green'], edgecolor='black')
        axes[1, 1].set_ylabel('孔隙率')
        axes[1, 1].set_title('孔隙率对比')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "statistics.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 已保存: statistics.png")

    def save_data(self):
        """保存数据"""
        np.save(os.path.join(self.save_dir, 'volume.npy'), self.volume)
        np.save(os.path.join(self.save_dir, 'gap_mask.npy'), self.gap_mask)
        np.save(os.path.join(self.save_dir, 'gap_regions.npy'), self.gap_regions)

        results = {
            'vol_size': self.vol_size,
            'target_porosity': float(self.target_porosity),
            'actual_porosity': float(self.gap_mask.sum() / self.gap_mask.size),
            'n_particles': len(self.particles),
            'n_gap_regions': int(self.gap_regions.max()),
            'blur_mean': self.blur_mean,
            'blur_std': self.blur_std
        }

        with open(os.path.join(self.save_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)

        print("✓ 已保存所有数据")


# ===================== 4. 3D GAN网络（内存优化版） =====================
class Generator3D(nn.Module):
    """3D生成器（内存优化：使用更小的特征维度）"""

    def __init__(self, latent_dim=64, vol_size=32):  # 减小latent_dim和默认尺寸
        super().__init__()
        self.latent_dim = latent_dim

        # 编码器（减少中间层大小）
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, 256),  # 从512减到256
            nn.ReLU(True),
            nn.Linear(256, 4 * 4 * 4 * 32)  # 从64减到32通道
        )

        # 主干网络（减少通道数）
        self.main = nn.ModuleList([
            # 4 -> 8
            nn.ConvTranspose3d(32, 64, 4, 2, 1),  # 从128减到64
            nn.BatchNorm3d(64),
            nn.ReLU(True),

            # 8 -> 16
            nn.ConvTranspose3d(64, 32, 4, 2, 1),  # 从64减到32
            nn.BatchNorm3d(32),
            nn.ReLU(True),

            # 16 -> 32
            nn.ConvTranspose3d(32, 16, 4, 2, 1),  # 从32减到16
            nn.BatchNorm3d(16),
            nn.ReLU(True),

            # 输出层
            nn.Conv3d(16, 1, 3, 1, 1),
            nn.Sigmoid()
        ])

        # 简化的注意力模块（可选）
        self.use_attention = vol_size <= 32  # 只在小尺寸时使用
        if self.use_attention:
            self.attention = SelfAttention3D(32)

    def forward(self, z):
        x = self.encoder(z).view(-1, 32, 4, 4, 4)

        for i, layer in enumerate(self.main):
            x = layer(x)
            # 只在特定层使用注意力，且尺寸不大时
            if self.use_attention and i == 5 and x.size(2) <= 16:
                x = self.attention(x)

        return x


class Discriminator3D(nn.Module):
    """3D判别器（内存优化版）"""

    def __init__(self, vol_size=32):
        super().__init__()

        self.main = nn.Sequential(
            # 32 -> 16
            nn.Conv3d(1, 16, 4, 2, 1),  # 从32减到16
            nn.LeakyReLU(0.2, True),
            nn.Dropout3d(0.2),  # 添加dropout减少过拟合

            # 16 -> 8
            nn.Conv3d(16, 32, 4, 2, 1),  # 从64减到32
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, True),
            nn.Dropout3d(0.2),

            # 8 -> 4
            nn.Conv3d(32, 64, 4, 2, 1),  # 从128减到64
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, True),
            nn.Dropout3d(0.2),

            # 4 -> 1
            nn.Conv3d(64, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)


class SelfAttention3D(nn.Module):
    """3D自注意力机制（内存优化：减少计算量）"""

    def __init__(self, channels):
        super().__init__()
        # 使用更大的缩减比例
        self.query = nn.Conv3d(channels, channels // 16, 1)  # 从//8改为//16
        self.key = nn.Conv3d(channels, channels // 16, 1)
        self.value = nn.Conv3d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        # 添加池化层减少空间维度
        self.pool = nn.AvgPool3d(2, 2)

    def forward(self, x):
        B, C, D, H, W = x.size()

        # 先池化减少空间维度
        x_pooled = self.pool(x)
        _, _, D_p, H_p, W_p = x_pooled.size()

        query = self.query(x_pooled).view(B, -1, D_p * H_p * W_p).permute(0, 2, 1)
        key = self.key(x_pooled).view(B, -1, D_p * H_p * W_p)
        value = self.value(x_pooled).view(B, -1, D_p * H_p * W_p)

        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, D_p, H_p, W_p)

        # 上采样回原始尺寸
        out = nn.functional.interpolate(out, size=(D, H, W), mode='trilinear', align_corners=False)

        return self.gamma * out + x


# ===================== 5. GAN训练器（内存优化版） =====================
class PorousGANTrainer:
    """GAN训练器（内存优化）"""

    def __init__(self, vol_size=32, latent_dim=64, device='cuda', save_dir='output_gan'):
        self.vol_size = vol_size
        self.latent_dim = latent_dim
        self.device = device
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)

        # 内存优化：如果GPU内存不足，自动切换到CPU
        try:
            self.G = Generator3D(latent_dim, vol_size).to(device)
            self.D = Discriminator3D(vol_size).to(device)

            # 测试前向传播
            test_z = torch.randn(1, latent_dim, device=device)
            _ = self.G(test_z)

            print(f"✓ 使用设备: {device}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️  GPU内存不足，切换到CPU模式")
                device = 'cpu'
                self.device = device
                torch.cuda.empty_cache()  # 清空GPU缓存

                self.G = Generator3D(latent_dim, vol_size).to(device)
                self.D = Discriminator3D(vol_size).to(device)
            else:
                raise e

        self.opt_G = optim.Adam(self.G.parameters(), lr=0.0001, betas=(0.5, 0.999))  # 降低学习率
        self.opt_D = optim.Adam(self.D.parameters(), lr=0.0001, betas=(0.5, 0.999))

        self.criterion = nn.BCELoss()

        self.history = {'loss_G': [], 'loss_D': []}

    def train(self, real_data_loader, epochs=200, accumulation_steps=4):
        """
        训练GAN（梯度累积优化）

        Parameters:
        -----------
        accumulation_steps : int
            梯度累积步数，用于模拟更大的batch_size
        """
        print(f"开始GAN训练 ({epochs} epochs, 设备: {self.device})")
        if self.device == 'cpu':
            print("💡 提示：CPU训练较慢，建议使用较少的epochs")

        for epoch in tqdm(range(epochs), desc="训练进度"):
            epoch_loss_D = 0
            epoch_loss_G = 0
            n_batches = 0

            for batch_idx, batch_data in enumerate(real_data_loader):
                # 解包数据
                if isinstance(batch_data, (list, tuple)):
                    real_batch = batch_data[0]
                else:
                    real_batch = batch_data

                batch_size = real_batch.size(0)
                real_batch = real_batch.to(self.device)

                # === 训练判别器 ===
                if batch_idx % accumulation_steps == 0:
                    self.opt_D.zero_grad()

                real_labels = torch.ones(batch_size, 1, device=self.device) * 0.9
                fake_labels = torch.zeros(batch_size, 1, device=self.device) + 0.1

                pred_real = self.D(real_batch)
                loss_D_real = self.criterion(pred_real, real_labels) / accumulation_steps

                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_batch = self.G(z)
                pred_fake = self.D(fake_batch.detach())
                loss_D_fake = self.criterion(pred_fake, fake_labels) / accumulation_steps

                loss_D = (loss_D_real + loss_D_fake)
                loss_D.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    self.opt_D.step()

                # === 训练生成器 ===
                if batch_idx % accumulation_steps == 0:
                    self.opt_G.zero_grad()

                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_batch = self.G(z)
                pred_fake = self.D(fake_batch)
                loss_G = self.criterion(pred_fake, torch.ones(batch_size, 1, device=self.device)) / accumulation_steps

                loss_G.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    self.opt_G.step()

                epoch_loss_D += loss_D.item() * accumulation_steps
                epoch_loss_G += loss_G.item() * accumulation_steps
                n_batches += 1

                # 内存清理
                if self.device == 'cuda' and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

            avg_loss_D = epoch_loss_D / n_batches
            avg_loss_G = epoch_loss_G / n_batches

            self.history['loss_G'].append(avg_loss_G)
            self.history['loss_D'].append(avg_loss_D)

            if (epoch + 1) % 20 == 0:
                self.save_samples(epoch + 1)
                print(f"\nEpoch {epoch + 1}: Loss_D={avg_loss_D:.4f}, Loss_G={avg_loss_G:.4f}")

        self.save_model()

        # 绘制训练曲线
        self._plot_training_curves()

    def _plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 5))

        plt.plot(self.history['loss_G'], label='Generator Loss', linewidth=2)
        plt.plot(self.history['loss_D'], label='Discriminator Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GAN Training Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 已保存训练曲线")

    def save_samples(self, epoch):
        """保存生成样本"""
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(1, self.latent_dim, device=self.device)
            sample = self.G(z).cpu().numpy()[0, 0]

        np.save(os.path.join(self.save_dir, 'samples', f'sample_epoch_{epoch}.npy'), sample)
        self.G.train()

    def save_model(self):
        """保存模型"""
        torch.save({
            'generator': self.G.state_dict(),
            'discriminator': self.D.state_dict(),
            'history': self.history
        }, os.path.join(self.save_dir, 'model.pth'))
        print("✓ 已保存模型")

    def generate_batch(self, n_samples=50):
        """生成批量样本（批量处理以节省内存）"""
        self.G.eval()
        samples = []

        # 批量生成（每次生成少量以节省内存）
        batch_size = 5 if self.device == 'cuda' else 1
        n_batches = (n_samples + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(range(n_batches), desc="生成样本"):
                current_batch_size = min(batch_size, n_samples - i * batch_size)
                z = torch.randn(current_batch_size, self.latent_dim, device=self.device)
                batch_samples = self.G(z).cpu().numpy()

                for j in range(current_batch_size):
                    sample = batch_samples[j, 0]
                    samples.append(sample)

                    sample_idx = i * batch_size + j
                    np.save(os.path.join(self.save_dir, 'samples',
                                         f'generated_{sample_idx:03d}.npy'), sample)

                # 清理内存
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

        return samples


# ===================== 6. 对比分析 =====================
def compare_methods(blurred_data, gan_samples, save_dir='comparison'):
    """对比模糊边缘法和GAN生成结果"""
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 80)
    print("对比分析: 模糊边缘法 vs GAN生成")
    print("=" * 80)

    # 1. 孔隙率对比
    blurred_porosity = blurred_data.sum() / blurred_data.size
    gan_porosities = [s.sum() / s.size for s in gan_samples]

    print(f"\n孔隙率统计:")
    print(f"  模糊边缘法: {blurred_porosity:.6f}")
    print(f"  GAN平均: {np.mean(gan_porosities):.6f} ± {np.std(gan_porosities):.6f}")

    # 2. 缝隙连通性分析
    blurred_connectivity = analyze_connectivity(blurred_data)
    gan_connectivities = [analyze_connectivity(s > 0.5) for s in gan_samples]

    print(f"\n连通性统计:")
    print(f"  模糊边缘法: {blurred_connectivity:.3f}")
    print(f"  GAN平均: {np.mean(gan_connectivities):.3f} ± {np.std(gan_connectivities):.3f}")

    # 3. 可视化对比
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 模糊边缘法中间切片
    mid = blurred_data.shape[0] // 2
    axes[0, 0].imshow(blurred_data[mid], cmap='gray')
    axes[0, 0].set_title(f'模糊边缘法 (孔隙率={blurred_porosity:.6f})')

    # GAN样本
    for i in range(5):
        ax = axes.flat[i + 1]
        sample = gan_samples[i]
        mid = sample.shape[0] // 2
        ax.imshow(sample[mid], cmap='gray')
        ax.set_title(f'GAN样本{i + 1} (孔隙率={gan_porosities[i]:.6f})')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison.png'), dpi=300)
    plt.close()

    # 4. 统计曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(gan_porosities, bins=20, alpha=0.7, label='GAN', color='blue', edgecolor='black')
    axes[0].axvline(blurred_porosity, color='red', linestyle='--', linewidth=2, label='模糊边缘法')
    axes[0].set_xlabel('孔隙率')
    axes[0].set_ylabel('频数')
    axes[0].set_title('孔隙率分布对比')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(gan_connectivities, bins=20, alpha=0.7, label='GAN', color='green', edgecolor='black')
    axes[1].axvline(blurred_connectivity, color='red', linestyle='--', linewidth=2, label='模糊边缘法')
    axes[1].set_xlabel('连通性系数')
    axes[1].set_ylabel('频数')
    axes[1].set_title('连通性分布对比')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statistics_comparison.png'), dpi=300)
    plt.close()

    print("\n✓ 对比分析完成")


def analyze_connectivity(volume):
    """分析连通性"""
    labeled, n_components = measure.label(volume > 0.5, connectivity=3, return_num=True)

    if n_components == 0:
        return 0.0

    # 最大连通分量占比
    largest_component_size = max([np.sum(labeled == i) for i in range(1, n_components + 1)])
    total_pore_voxels = np.sum(volume > 0.5)

    if total_pore_voxels == 0:
        return 0.0

    return largest_component_size / total_pore_voxels


# ===================== 7. 主程序 =====================
if __name__ == "__main__":
    # 第1步：体素容量分析
    print("第1步：理论分析")
    analyzer = VoxelCapacityAnalyzer()
    capacity_results = analyzer.analyze_capacity([256, 512])

    # 第2步：模糊边缘法生成
    print("\n第2步：模糊边缘法生成")

    # 针对极低孔隙率优化的参数配置
    print("\n💡 极低孔隙率(0.0003)特殊配置：")
    print("  策略：致密堆积 + 颗粒边界缝隙")
    print("  - 大量小颗粒填充空间")
    print("  - 仅在颗粒接触面产生微小缝隙")
    print("  - 使用概率控制缝隙生成")
    print()

    blurred_method = BlurredEdgeMethod(
        vol_size=128,  # 足够大的空间
        target_porosity=0.0003,
        particle_diameter=5,  # 小颗粒，高密度
        compactness=0.99,  # 极高密实度
        blur_mean=1.0,  # 小边缘模糊（缝隙宽度）
        blur_std=0.5,  # 小方差
        save_dir='output_blurred'
    )

    blurred_method.generate_particles()
    blurred_method.create_volume_with_blurred_edges()

    # 检查孔隙率偏差
    actual_porosity = blurred_method.gap_mask.sum() / blurred_method.gap_mask.size
    relative_error = abs(actual_porosity - 0.0003) / 0.0003

    if relative_error > 0.5:  # 相对误差>50%
        print("\n" + "=" * 80)
        print("⚠️  孔隙率偏差分析")
        print("=" * 80)
        print(f"目标孔隙率: {0.0003:.6f}")
        print(f"实际孔隙率: {actual_porosity:.6f}")
        print(f"绝对偏差: {abs(actual_porosity - 0.0003):.6f}")
        print(f"相对误差: {relative_error * 100:.1f}%")

        print("\n原因分析:")
        print("  1. 极低孔隙率(0.03%)接近材料物理极限")
        print("  2. 颗粒间必然存在几何间隙（不可完全消除）")
        print("  3. 边缘模糊算法会在颗粒边界产生过渡区")

        print("\n改进建议:")
        if actual_porosity > 0.0003:
            suggestions = [
                ("增加颗粒数量",
                 f"当前: {len(blurred_method.particles)}, 建议: {int(len(blurred_method.particles) * 1.5)}"),
                ("减小颗粒直径", f"当前: 5体素, 建议: 4体素"),
                ("提高密实度", f"当前: 0.99, 建议: 0.995"),
                ("减小缝隙宽度", f"当前blur_mean: 2, 建议: 1.5"),
                ("增大体素空间", f"当前: 128³, 建议: 256³")
            ]
        else:
            suggestions = [
                ("减少颗粒数量", f"当前: {len(blurred_method.particles)}"),
                ("增大颗粒直径", f"当前: 5体素, 建议: 6体素"),
                ("降低密实度", f"当前: 0.99, 建议: 0.95")
            ]

        for i, (action, detail) in enumerate(suggestions, 1):
            print(f"  {i}. {action}: {detail}")

        print("\n💡 提示：对于如此极端的低孔隙率，建议使用更大的体素空间")
        print("   (如256³或512³)以获得更精确的控制。")
        print("=" * 80 + "\n")
    else:
        print(f"\n✓ 孔隙率控制良好，相对误差: {relative_error * 100:.1f}%\n")

    blurred_method.extract_gap_regions()
    blurred_method.find_widest_path()
    blurred_method.visualize_results()
    blurred_method.save_data()

    # 第3步：准备GAN训练数据
    print("\n第3步：准备GAN训练数据")
    blurred_volume = blurred_method.volume

    # 创建数据加载器
    from torch.utils.data import TensorDataset, DataLoader

    real_data = torch.FloatTensor(blurred_volume).unsqueeze(0).unsqueeze(0)

    # 数据增强：创建多个旋转和翻转版本
    augmented_data = [real_data]

    # 90度旋转（3个方向，每个方向3次旋转）
    for axis in [2, 3, 4]:  # Z, Y, X轴
        for k in [1, 2, 3]:
            if axis == 2:  # Z轴
                augmented_data.append(torch.rot90(real_data, k=k, dims=[3, 4]))
            elif axis == 3:  # Y轴
                augmented_data.append(torch.rot90(real_data, k=k, dims=[2, 4]))
            else:  # X轴
                augmented_data.append(torch.rot90(real_data, k=k, dims=[2, 3]))

    # 翻转
    augmented_data.append(torch.flip(real_data, dims=[2]))
    augmented_data.append(torch.flip(real_data, dims=[3]))
    augmented_data.append(torch.flip(real_data, dims=[4]))

    print(f"数据增强后样本数: {len(augmented_data)}")

    dataset = TensorDataset(torch.cat(augmented_data))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 第4步：GAN训练（内存优化）
    print("\n第4步：GAN训练（内存优化版）")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 根据可用内存自动调整参数
    if device == 'cuda':
        # 检测GPU内存
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
        print(f"GPU内存: {gpu_mem:.2f} GB")

        if gpu_mem < 6:
            gan_vol_size = 24  # 超小尺寸
            batch_size = 1
            latent_dim = 32
            print("⚠️  GPU内存较小(<6GB)，使用超小网络配置")
        elif gpu_mem < 10:
            gan_vol_size = 32  # 小尺寸
            batch_size = 2
            latent_dim = 64
            print("💡 GPU内存中等(6-10GB)，使用小网络配置")
        else:
            gan_vol_size = 48  # 中等尺寸
            batch_size = 4
            latent_dim = 64
            print("✓ GPU内存充足(>10GB)，使用中等网络配置")
    else:
        gan_vol_size = 24
        batch_size = 1
        latent_dim = 32
        print("💻 使用CPU训练（较慢）")

    print(f"\n配置参数:")
    print(f"  - GAN生成尺寸: {gan_vol_size}³")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Latent dim: {latent_dim}")
    print(f"  - 训练轮数: 50 (减少以节省时间)")

    # 下采样真实数据以匹配GAN尺寸
    from scipy.ndimage import zoom

    scale_factor = gan_vol_size / blurred_method.vol_size
    print(f"  - 下采样比例: {scale_factor:.3f}")

    downsampled_volume = zoom(blurred_volume, scale_factor, order=1)

    downsampled_data = torch.FloatTensor(downsampled_volume).unsqueeze(0).unsqueeze(0)

    # 简化的数据增强（减少内存使用）
    augmented_downsampled = [downsampled_data]

    # 只做90度旋转，不做所有角度
    for k in [1, 2, 3]:
        augmented_downsampled.append(torch.rot90(downsampled_data, k=k, dims=[3, 4]))

    # 翻转
    augmented_downsampled.append(torch.flip(downsampled_data, dims=[2]))

    print(f"  - 增强后样本数: {len(augmented_downsampled)}")

    dataset_gan = TensorDataset(torch.cat(augmented_downsampled))
    dataloader_gan = DataLoader(dataset_gan, batch_size=batch_size, shuffle=True,
                                pin_memory=False)  # 关闭pin_memory节省内存

    gan_trainer = PorousGANTrainer(vol_size=gan_vol_size, latent_dim=latent_dim,
                                   device=device, save_dir='output_gan')

    # 使用更少的epochs
    gan_trainer.train(dataloader_gan, epochs=50, accumulation_steps=2)

    # 第5步：生成50幅图像
    print("\n第5步：生成批量图像")
    gan_samples = gan_trainer.generate_batch(n_samples=50)

    # 第6步：对比分析
    print("\n第6步：对比分析")
    # 下采样真实数据用于对比
    downsampled_mask = zoom(blurred_method.gap_mask.astype(float), scale_factor, order=0)
    compare_methods(downsampled_mask, gan_samples, save_dir='comparison')

    print("\n" + "=" * 80)
    print("所有步骤完成！")
    print("=" * 80)
    print("\n📊 结果总结:")
    print(f"  1. 模糊边缘法:")
    print(f"     - 体素空间: {blurred_method.vol_size}³")
    print(f"     - 生成颗粒数: {len(blurred_method.particles)}")
    print(f"     - 实际孔隙率: {actual_porosity:.6f}")
    print(f"     - 缝隙区域数: {blurred_method.gap_regions.max()}")
    print(f"\n  2. GAN生成:")
    print(f"     - 生成样本数: 50")
    print(f"     - 训练轮数: 100")
    print(f"     - 生成尺寸: {gan_vol_size}³")
    print(f"\n📁 结果保存位置:")
    print(f"  - output_blurred/: 模糊边缘法结果")
    print(f"  - output_gan/: GAN训练结果和生成样本")
    print(f"  - comparison/: 对比分析结果")
    print("=" * 80)