import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage import morphology, measure
from skimage.filters import sobel
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import networkx as nx
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


# ==================== GAN架构 ====================
class AttentionBlock(nn.Module):
    """注意力机制模块"""

    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)

        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        return self.gamma * out + x


class Generator(nn.Module):
    """带多尺度特征和注意力的生成器"""

    def __init__(self, latent_dim=100, img_size=256):
        super().__init__()
        self.init_size = img_size // 16
        self.l1 = nn.Linear(latent_dim, 512 * self.init_size ** 2)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            AttentionBlock(256),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            AttentionBlock(64),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.size(0), 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    """带多尺度特征的判别器"""

    def __init__(self, img_size=256):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
        )

        ds_size = img_size // 16
        self.adv_layer = nn.Sequential(
            nn.Linear(512 * ds_size ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.size(0), -1)
        validity = self.adv_layer(out)
        return validity


# ==================== Voronoi颗粒生成 ====================
class VoronoiParticleGenerator:
    """Voronoi颗粒生成器"""

    def __init__(self, img_size=256):
        self.img_size = img_size

    def calculate_max_particles(self, img_size, min_diameter=3, packing_ratio=0.65):
        """
        计算最大允许颗粒数量

        参数:
        - img_size: 图像尺寸
        - min_diameter: 最小颗粒直径(像素)
        - packing_ratio: 有效堆积率(0-1)
          * 0.65 (默认): 孔隙率35%, 适合松散堆积
          * 0.70-0.75: 孔隙率25-30%, 适合中等密实
          * 0.75-0.80: 孔隙率20-25%, 适合密实堆积
          * 0.50-0.60: 孔隙率40-50%, 适合高孔隙材料

        返回: 最大颗粒数量
        """
        area = img_size ** 2
        min_particle_area = np.pi * (min_diameter / 2) ** 2
        # 考虑间隙,实际颗粒面积占比由packing_ratio决定
        max_particles = int(area * packing_ratio / min_particle_area)
        return max_particles

    def generate_particle_diameters(self, n_particles, mean_diameter, std_ratio=0.3,
                                    distribution='lognormal'):
        """生成符合地质统计学规律的颗粒直径分布"""
        if distribution == 'lognormal':
            # 对数正态分布 - 常见于沉积物
            sigma = std_ratio
            mu = np.log(mean_diameter) - 0.5 * sigma ** 2
            diameters = np.random.lognormal(mu, sigma, n_particles)
        elif distribution == 'gamma':
            # Gamma分布 - 适用于某些地质材料
            k = (1 / std_ratio) ** 2
            theta = mean_diameter / k
            diameters = np.random.gamma(k, theta, n_particles)
        elif distribution == 'weibull':
            # Weibull分布 - 适用于破碎材料
            k = 2.5
            lambda_ = mean_diameter / (np.gamma(1 + 1 / k))
            diameters = lambda_ * np.random.weibull(k, n_particles)
        else:
            # 正态分布
            diameters = np.random.normal(mean_diameter, mean_diameter * std_ratio, n_particles)

        # 限制直径范围
        diameters = np.clip(diameters, mean_diameter * 0.3, mean_diameter * 3)
        return diameters

    def generate_polygon_sides(self, n_particles, min_sides=3, max_sides=100):
        """生成多边形边数(3-100边形)"""
        # 使用加权分布,较少边数的多边形更常见
        weights = 1.0 / np.arange(min_sides, max_sides + 1)
        weights = weights / weights.sum()
        sides = np.random.choice(range(min_sides, max_sides + 1),
                                 size=n_particles, p=weights)
        return sides

    def create_voronoi_with_gaps(self, n_particles, porosity=0.3,
                                 mean_diameter=None, distribution='lognormal'):
        """创建带间隙的Voronoi颗粒图"""
        if mean_diameter is None:
            # 根据颗粒数和孔隙率估算平均直径
            effective_area = self.img_size ** 2 * (1 - porosity)
            mean_diameter = np.sqrt(effective_area / n_particles / np.pi) * 2

        # 生成颗粒直径
        diameters = self.generate_particle_diameters(n_particles, mean_diameter,
                                                     distribution=distribution)

        # 生成多边形边数
        polygon_sides = self.generate_polygon_sides(n_particles)

        # 生成Voronoi点
        points = np.random.rand(n_particles, 2) * self.img_size
        vor = Voronoi(points)

        # 创建图像
        image = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        particle_masks = []

        # 为每个颗粒创建多边形
        for idx, (point, diameter, n_sides) in enumerate(zip(points, diameters, polygon_sides)):
            radius = diameter / 2
            # 创建多边形顶点
            angles = np.linspace(0, 2 * np.pi, n_sides + 1)[:-1]
            # 添加随机扰动使其更不规则
            angles += np.random.randn(n_sides) * 0.1
            radii = radius * (1 + np.random.randn(n_sides) * 0.1)

            vertices_x = point[0] + radii * np.cos(angles)
            vertices_y = point[1] + radii * np.sin(angles)

            # 创建颗粒掩码
            mask = np.zeros((self.img_size, self.img_size), dtype=bool)
            y, x = np.ogrid[:self.img_size, :self.img_size]

            # 使用多边形填充
            for i in range(n_sides):
                x1, y1 = vertices_x[i], vertices_y[i]
                x2, y2 = vertices_x[(i + 1) % n_sides], vertices_y[(i + 1) % n_sides]

                # 简化的多边形填充
                center_x, center_y = point[0], point[1]
                dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                mask |= dist_from_center < radius * 0.9  # 稍微缩小以产生间隙

            particle_masks.append(mask)
            image[mask] = 1

        # 应用腐蚀以创建间隙
        gap_size = max(1, int(mean_diameter * 0.05))
        image = morphology.binary_erosion(image, morphology.disk(gap_size))

        # 模糊边缘
        image = gaussian_filter(image.astype(float), sigma=1.5)

        return image, particle_masks, diameters, polygon_sides


# ==================== 孔隙分析 ====================
class PoreAnalyzer:
    """孔隙路径分析器"""

    def __init__(self, image):
        self.image = image
        self.pore_mask = image < 0.5  # 孔隙区域

    def find_all_gaps(self):
        """找到所有缝隙并标记"""
        # 标记连通区域
        labeled_pores = measure.label(self.pore_mask)
        n_pores = labeled_pores.max()

        # 为每个缝隙分配不同颜色
        gap_colored = np.zeros((*self.image.shape, 3))
        colors = plt.cm.rainbow(np.linspace(0, 1, n_pores))

        gap_info = []
        for i in range(1, n_pores + 1):
            mask = labeled_pores == i
            area = mask.sum()
            gap_colored[mask] = colors[i - 1][:3]
            gap_info.append({
                'id': i,
                'area': area,
                'mask': mask
            })

        return gap_colored, gap_info

    def find_widest_path(self):
        """找到最宽的连通路径"""
        # 计算距离变换(到最近固体的距离)
        distance = distance_transform_edt(self.pore_mask)

        # 找到最宽的路径
        skeleton = morphology.skeletonize(self.pore_mask)
        widths = distance[skeleton]

        if len(widths) == 0:
            return None, 0

        max_width = widths.max() * 2  # 直径

        # 标记最宽路径
        widest_path = np.zeros_like(self.image)
        threshold = max_width * 0.8
        widest_path[distance >= threshold / 2] = 1

        return widest_path, max_width

    def trace_pore_paths(self):
        """追踪孔隙路径"""
        skeleton = morphology.skeletonize(self.pore_mask)

        # 使用图论追踪路径
        y_coords, x_coords = np.where(skeleton)

        if len(y_coords) == 0:
            return skeleton

        # 简化的路径表示
        path_image = skeleton.astype(float)
        path_image = gaussian_filter(path_image, sigma=1.0)

        return path_image


# ==================== 地质统计分析 ====================
class GeostatisticalAnalyzer:
    """地质统计学分析器"""

    def __init__(self, image):
        self.image = image
        self.porosity = (image < 0.5).sum() / image.size

    def calculate_porosity(self):
        """计算孔隙率"""
        return self.porosity

    def particle_size_distribution(self, diameters):
        """颗粒粒径分布分析"""
        stats_dict = {
            'mean': np.mean(diameters),
            'std': np.std(diameters),
            'cv': np.std(diameters) / np.mean(diameters),  # 变异系数
            'd10': np.percentile(diameters, 10),
            'd50': np.percentile(diameters, 50),
            'd90': np.percentile(diameters, 90),
            'uniformity': np.percentile(diameters, 60) / np.percentile(diameters, 10),
            'curvature': np.percentile(diameters, 30) ** 2 / (
                        np.percentile(diameters, 10) * np.percentile(diameters, 60))
        }
        return stats_dict

    def spatial_correlation(self):
        """空间相关性分析 - 半变异函数"""
        binary = (self.image > 0.5).astype(int)
        h_max = min(50, self.image.shape[0] // 4)
        lags = range(1, h_max, 2)
        variogram = []

        for h in lags:
            diff_h = binary[h:, :] - binary[:-h, :]
            gamma_h = 0.5 * np.mean(diff_h ** 2)
            variogram.append(gamma_h)

        return lags, variogram

    def percolation_analysis(self):
        """渗透性分析"""
        pore_mask = self.image < 0.5
        labeled = measure.label(pore_mask)

        # 检查是否存在跨越样本的连通路径
        top_labels = set(labeled[0, :])
        bottom_labels = set(labeled[-1, :])
        percolating_labels = top_labels & bottom_labels
        percolating_labels.discard(0)

        percolates = len(percolating_labels) > 0

        if percolates:
            largest_cluster = max(percolating_labels,
                                  key=lambda x: (labeled == x).sum())
            percolation_ratio = (labeled == largest_cluster).sum() / pore_mask.sum()
        else:
            percolation_ratio = 0

        return percolates, percolation_ratio

    def hydraulic_conductivity_estimate(self, diameters):
        """估算水力传导度(基于Kozeny-Carman方程)"""
        porosity = self.porosity
        d50 = np.percentile(diameters, 50)

        # Kozeny-Carman方程
        k = (d50 ** 2 * porosity ** 3) / (180 * (1 - porosity) ** 2)

        return k


# ==================== 主系统 ====================
class PorousMediaGenerator:
    """多孔材料生成主系统"""

    def __init__(self, img_size=256):
        self.img_size = img_size
        self.voronoi_gen = VoronoiParticleGenerator(img_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def generate(self, n_particles, porosity=0.3, mean_diameter=None,
                 distribution='lognormal'):
        """生成多孔材料图像"""
        # 生成基础Voronoi图
        image, masks, diameters, polygon_sides = self.voronoi_gen.create_voronoi_with_gaps(
            n_particles, porosity, mean_diameter, distribution
        )

        # 孔隙分析
        analyzer = PoreAnalyzer(image)
        gap_colored, gap_info = analyzer.find_all_gaps()
        widest_path, max_width = analyzer.find_widest_path()
        pore_paths = analyzer.trace_pore_paths()

        # 地质统计分析
        geo_analyzer = GeostatisticalAnalyzer(image)
        porosity_actual = geo_analyzer.calculate_porosity()
        particle_stats = geo_analyzer.particle_size_distribution(diameters)
        percolates, perc_ratio = geo_analyzer.percolation_analysis()
        hydraulic_k = geo_analyzer.hydraulic_conductivity_estimate(diameters)

        results = {
            'image': image,
            'gap_colored': gap_colored,
            'widest_path': widest_path,
            'pore_paths': pore_paths,
            'max_width': max_width,
            'porosity': porosity_actual,
            'n_gaps': len(gap_info),
            'particle_stats': particle_stats,
            'diameters': diameters,
            'polygon_sides': polygon_sides,
            'percolates': percolates,
            'percolation_ratio': perc_ratio,
            'hydraulic_conductivity': hydraulic_k
        }

        return results


# ==================== 可视化和测试 ====================
def visualize_results(results, img_size):
    """可视化结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 原始颗粒图
    axes[0, 0].imshow(results['image'], cmap='gray')
    axes[0, 0].set_title(f'Porous Media ({img_size}x{img_size})\nPorosity: {results["porosity"]:.3f}')
    axes[0, 0].axis('off')

    # 缝隙标记
    axes[0, 1].imshow(results['gap_colored'])
    axes[0, 1].set_title(f'Gap Identification\n{results["n_gaps"]} gaps found')
    axes[0, 1].axis('off')

    # 最宽路径
    axes[0, 2].imshow(results['image'], cmap='gray', alpha=0.5)
    if results['widest_path'] is not None:
        axes[0, 2].imshow(results['widest_path'], cmap='Reds', alpha=0.5)
    axes[0, 2].set_title(f'Widest Path\nMax width: {results["max_width"]:.2f}px')
    axes[0, 2].axis('off')

    # 孔隙路径
    axes[1, 0].imshow(results['image'], cmap='gray', alpha=0.5)
    axes[1, 0].imshow(results['pore_paths'], cmap='jet', alpha=0.6)
    axes[1, 0].set_title('Pore Path Network')
    axes[1, 0].axis('off')

    # 颗粒直径分布
    axes[1, 1].hist(results['diameters'], bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Particle Diameter Distribution')
    axes[1, 1].set_xlabel('Diameter (pixels)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)

    # 多边形边数分布
    sides_unique, counts = np.unique(results['polygon_sides'], return_counts=True)
    axes[1, 2].bar(sides_unique, counts, edgecolor='black', alpha=0.7)
    axes[1, 2].set_title('Polygon Sides Distribution')
    axes[1, 2].set_xlabel('Number of Sides')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def print_analysis_results(results):
    """打印分析结果"""
    print("\n" + "=" * 60)
    print("GEOSTATISTICAL ANALYSIS RESULTS")
    print("=" * 60)

    print(f"\nPorosity: {results['porosity']:.4f}")
    print(f"Number of gaps: {results['n_gaps']}")
    print(f"Maximum gap width: {results['max_width']:.2f} pixels")
    print(f"Percolation: {'Yes' if results['percolates'] else 'No'}")
    if results['percolates']:
        print(f"Percolation ratio: {results['percolation_ratio']:.4f}")
    print(f"Hydraulic conductivity estimate: {results['hydraulic_conductivity']:.2e}")

    print("\nParticle Size Statistics:")
    stats = results['particle_stats']
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

    print("\nPolygon Distribution:")
    sides_unique, counts = np.unique(results['polygon_sides'], return_counts=True)
    print(f"  Range: {sides_unique.min()}-{sides_unique.max()} sides")
    print(f"  Most common: {sides_unique[counts.argmax()]} sides ({counts.max()} particles)")


def calculate_max_particles_table():
    """计算不同图像尺寸的最大颗粒数量"""
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    print("\n" + "=" * 80)
    print("MAXIMUM PARTICLE CAPACITY (不同堆积率对比)")
    print("=" * 80)

    packing_ratios = [0.50, 0.65, 0.75]
    packing_names = ['松散(孔隙率50%)', '中等(孔隙率35%)', '密实(孔隙率25%)']

    for ratio, name in zip(packing_ratios, packing_names):
        print(f"\n{name} - Packing Ratio = {ratio:.2f}")
        print("-" * 80)
        print(f"{'图像尺寸':<15} {'最大颗粒数':<15} {'单颗粒面积':<20} {'实际孔隙率':<15}")
        print("-" * 80)

        for size in sizes:
            gen = VoronoiParticleGenerator(size)
            max_particles = gen.calculate_max_particles(size, min_diameter=3, packing_ratio=ratio)
            area_per_particle = (size * size) / max_particles
            actual_porosity = 1 - ratio
            print(f"{size}x{size:<10} {max_particles:<15} {area_per_particle:<20.2f} {actual_porosity:.1%}")

    print("\n" + "=" * 80)
    print("说明:")
    print("  - 基于最小颗粒直径3像素")
    print("  - 堆积率 = 颗粒面积/总面积")
    print("  - 孔隙率 = 1 - 堆积率 = 孔隙面积/总面积")
    print("  - 天然砂土孔隙率通常在25-40%之间")


def plot_max_particles_vs_size():
    """绘制最大颗粒数量与图像尺寸的关系曲线（组合图）"""
    # 生成更密集的尺寸点
    sizes = np.linspace(16, 1024, 100)

    packing_ratios = [0.50, 0.65, 0.75, 0.85]
    packing_names = ['松散 (φ=50%)', '中等 (φ=35%)', '密实 (φ=25%)', '极密实 (φ=15%)']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    markers = ['o', 's', '^', 'D']

    # 预计算所有数据
    all_data = {}
    for ratio in packing_ratios:
        max_particles_list = []
        density_list = []
        for size in sizes:
            gen = VoronoiParticleGenerator(int(size))
            max_p = gen.calculate_max_particles(int(size), min_diameter=3, packing_ratio=ratio)
            max_particles_list.append(max_p)
            density = max_p / (size ** 2)
            density_list.append(density * 1000)
        all_data[ratio] = {'particles': max_particles_list, 'density': density_list}

    # 创建组合图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 子图1: 线性坐标 - 最大颗粒数 vs 图像尺寸
    ax1 = axes[0, 0]
    for ratio, name, color, marker in zip(packing_ratios, packing_names, colors, markers):
        max_particles_list = all_data[ratio]['particles']

        ax1.plot(sizes, max_particles_list, label=name, color=color, linewidth=2.5)
        # 标记关键点
        key_sizes = [16, 64, 128, 256, 512, 1024]
        key_indices = [np.argmin(np.abs(sizes - s)) for s in key_sizes]
        ax1.scatter([sizes[i] for i in key_indices],
                    [max_particles_list[i] for i in key_indices],
                    color=color, s=80, marker=marker, zorder=5)

    ax1.set_xlabel('图像尺寸 (pixels)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('最大颗粒数量', fontsize=12, fontweight='bold')
    ax1.set_title('最大颗粒数量 vs 图像尺寸 (线性坐标)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([0, 1050])

    # 子图2: 对数坐标 - 最大颗粒数 vs 图像尺寸
    ax2 = axes[0, 1]
    for ratio, name, color, marker in zip(packing_ratios, packing_names, colors, markers):
        max_particles_list = all_data[ratio]['particles']

        ax2.loglog(sizes, max_particles_list, label=name, color=color, linewidth=2.5)
        key_sizes = [16, 64, 256, 1024]
        key_indices = [np.argmin(np.abs(sizes - s)) for s in key_sizes]
        ax2.scatter([sizes[i] for i in key_indices],
                    [max_particles_list[i] for i in key_indices],
                    color=color, s=80, marker=marker, zorder=5)

    ax2.set_xlabel('图像尺寸 (pixels)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('最大颗粒数量', fontsize=12, fontweight='bold')
    ax2.set_title('最大颗粒数量 vs 图像尺寸 (对数坐标)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--', which='both')

    # 添加理论曲线 N ∝ size²
    theoretical = (sizes / sizes[0]) ** 2 * 17.8  # 归一化到第一个点
    ax2.plot(sizes, theoretical, 'k--', linewidth=2, alpha=0.5, label='理论: N ∝ size²')
    ax2.legend(fontsize=10, loc='upper left')

    # 子图3: 颗粒密度 vs 图像尺寸
    ax3 = axes[1, 0]
    for ratio, name, color, marker in zip(packing_ratios, packing_names, colors, markers):
        density_list = all_data[ratio]['density']

        ax3.plot(sizes, density_list, label=name, color=color, linewidth=2.5)
        key_sizes = [16, 128, 512, 1024]
        key_indices = [np.argmin(np.abs(sizes - s)) for s in key_sizes]
        ax3.scatter([sizes[i] for i in key_indices],
                    [density_list[i] for i in key_indices],
                    color=color, s=80, marker=marker, zorder=5)

    ax3.set_xlabel('图像尺寸 (pixels)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('颗粒密度 (颗粒/1000像素²)', fontsize=12, fontweight='bold')
    ax3.set_title('颗粒密度 vs 图像尺寸', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim([0, 1050])

    # 子图4: 数据表格
    ax4 = axes[1, 1]
    ax4.axis('off')

    # 创建数据表格
    table_data = []
    table_data.append(['图像尺寸', '松散\n(φ=50%)', '中等\n(φ=35%)', '密实\n(φ=25%)', '极密实\n(φ=15%)'])

    key_sizes_table = [16, 32, 64, 128, 256, 512, 1024]
    for size in key_sizes_table:
        row = [f'{size}×{size}']
        for ratio in packing_ratios:
            gen = VoronoiParticleGenerator(size)
            max_p = gen.calculate_max_particles(size, min_diameter=3, packing_ratio=ratio)
            row.append(f'{max_p:,}')
        table_data.append(row)

    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # 设置表头样式
    for i in range(5):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 设置行颜色交替
    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')

    ax4.set_title('最大颗粒数量对照表', fontsize=14, fontweight='bold', pad=20)

    # 添加总标题和说明
    fig.suptitle('最大允许颗粒数量 vs 图像尺寸关系分析',
                 fontsize=16, fontweight='bold', y=0.995)

    fig.text(0.5, 0.02,
             '说明: φ=孔隙率, 基于最小颗粒直径3像素 | 理论关系: 最大颗粒数 ∝ (图像尺寸)²',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0.03, 1, 0.99])

    return fig, sizes, all_data


def plot_individual_figures(sizes, all_data):
    """分别绘制四个独立的图表"""
    packing_ratios = [0.50, 0.65, 0.75, 0.85]
    packing_names = ['φ=50%', 'φ=35%', 'φ=25%', 'φ=15%']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    markers = ['o', 's', '^', 'D']

    # 图1: 线性坐标
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    for ratio, name, color, marker in zip(packing_ratios, packing_names, colors, markers):
        max_particles_list = all_data[ratio]['particles']
        ax1.plot(sizes, max_particles_list, label=name, color=color, linewidth=3)
        key_sizes = [16, 64, 128, 256, 512, 1024]
        key_indices = [np.argmin(np.abs(sizes - s)) for s in key_sizes]
        ax1.scatter([sizes[i] for i in key_indices],
                    [max_particles_list[i] for i in key_indices],
                    color=color, s=100, marker=marker, zorder=5, edgecolors='black', linewidths=1.5)

    ax1.set_xlabel('Image size (pixels)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Maximum number of particles', fontsize=14, fontweight='bold')
    ax1.set_title('Maximum number of particles under different porosity levels', fontsize=16, fontweight='bold', pad=15)
    ax1.legend(fontsize=12, loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([0, 1050])
    plt.tight_layout()
    plt.savefig('figure1_linear_scale.png', dpi=300, bbox_inches='tight')
    print("图1已保存: figure1_linear_scale.png")
    plt.close()

    # 图2: 对数坐标
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    for ratio, name, color, marker in zip(packing_ratios, packing_names, colors, markers):
        max_particles_list = all_data[ratio]['particles']
        ax2.loglog(sizes, max_particles_list, label=name, color=color, linewidth=3)
        key_sizes = [16, 64, 256, 1024]
        key_indices = [np.argmin(np.abs(sizes - s)) for s in key_sizes]
        ax2.scatter([sizes[i] for i in key_indices],
                    [max_particles_list[i] for i in key_indices],
                    color=color, s=100, marker=marker, zorder=5, edgecolors='black', linewidths=1.5)

    # 添加理论曲线
    theoretical = (sizes / sizes[0]) ** 2 * 17.8
    ax2.plot(sizes, theoretical, 'k--', linewidth=2.5, alpha=0.6, label='理论: N ∝ size²')

    ax2.set_xlabel('图像尺寸 (pixels)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('最大颗粒数量', fontsize=14, fontweight='bold')
    ax2.set_title('最大颗粒数量 vs 图像尺寸 (对数坐标)', fontsize=16, fontweight='bold', pad=15)
    ax2.legend(fontsize=12, loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', which='both')
    plt.tight_layout()
    plt.savefig('figure2_log_scale.png', dpi=300, bbox_inches='tight')
    print("图2已保存: figure2_log_scale.png")
    plt.close()

    # 图3: 颗粒密度
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    for ratio, name, color, marker in zip(packing_ratios, packing_names, colors, markers):
        density_list = all_data[ratio]['density']
        ax3.plot(sizes, density_list, label=name, color=color, linewidth=3)
        key_sizes = [16, 128, 512, 1024]
        key_indices = [np.argmin(np.abs(sizes - s)) for s in key_sizes]
        ax3.scatter([sizes[i] for i in key_indices],
                    [density_list[i] for i in key_indices],
                    color=color, s=100, marker=marker, zorder=5, edgecolors='black', linewidths=1.5)

    ax3.set_xlabel('图像尺寸 (pixels)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('颗粒密度 (颗粒/1000像素²)', fontsize=14, fontweight='bold')
    ax3.set_title('颗粒密度 vs 图像尺寸', fontsize=16, fontweight='bold', pad=15)
    ax3.legend(fontsize=12, loc='upper right', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim([0, 1050])
    plt.tight_layout()
    plt.savefig('figure3_particle_density.png', dpi=300, bbox_inches='tight')
    print("图3已保存: figure3_particle_density.png")
    plt.close()

    # 图4: 数据表格
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    ax4.axis('off')

    table_data = []
    table_data.append(['图像尺寸', '松散\n(φ=50%)', '中等\n(φ=35%)', '密实\n(φ=25%)', '极密实\n(φ=15%)'])

    key_sizes_table = [16, 32, 64, 128, 256, 512, 1024]
    for size in key_sizes_table:
        row = [f'{size}×{size}']
        for ratio in packing_ratios:
            gen = VoronoiParticleGenerator(size)
            max_p = gen.calculate_max_particles(size, min_diameter=3, packing_ratio=ratio)
            row.append(f'{max_p:,}')
        table_data.append(row)

    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.18, 0.18, 0.18, 0.18, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)

    # 设置表头样式
    for i in range(5):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)

    # 设置行颜色交替
    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')

    ax4.set_title('最大颗粒数量对照表\n(基于最小颗粒直径3像素)',
                  fontsize=16, fontweight='bold', pad=30)

    # 添加说明文字
    fig4.text(0.5, 0.05,
              'φ = 孔隙率 | 堆积率 = 1 - 孔隙率 | 最大颗粒数 ∝ (图像尺寸)²',
              ha='center', fontsize=11, style='italic',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, pad=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig('figure4_data_table.png', dpi=300, bbox_inches='tight')
    print("图4已保存: figure4_data_table.png")
    plt.close()

def visualize_and_save_separately(results, img_size, save_prefix='analysis'):
    """分别保存每个分析图"""

    # 图1: 原始颗粒图
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.imshow(results['image'], cmap='gray')
    ax1.set_title(f'Porous Media ({img_size}x{img_size})\nPorosity: {results["porosity"]:.3f}',
                  fontsize=14, fontweight='bold')
    ax1.axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_1_porous_media.png', dpi=300, bbox_inches='tight')
    print(f"已保存: {save_prefix}_1_porous_media.png")
    plt.close()

    # 图2: 缝隙标记
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.imshow(results['gap_colored'])
    ax2.set_title(f'Gap Identification\n{results["n_gaps"]} gaps found',
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_2_gap_identification.png', dpi=300, bbox_inches='tight')
    print(f"已保存: {save_prefix}_2_gap_identification.png")
    plt.close()

    # 图3: 最宽路径
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    ax3.imshow(results['image'], cmap='gray', alpha=0.5)
    if results['widest_path'] is not None:
        ax3.imshow(results['widest_path'], cmap='Reds', alpha=0.5)
    ax3.set_title(f'Widest Path\nMax width: {results["max_width"]:.2f}px',
                  fontsize=14, fontweight='bold')
    ax3.axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_3_widest_path.png', dpi=300, bbox_inches='tight')
    print(f"已保存: {save_prefix}_3_widest_path.png")
    plt.close()

    # 图4: 孔隙路径
    fig4, ax4 = plt.subplots(figsize=(8, 8))
    ax4.imshow(results['image'], cmap='gray', alpha=0.5)
    ax4.imshow(results['pore_paths'], cmap='jet', alpha=0.6)
    ax4.set_title('Pore Path Network', fontsize=14, fontweight='bold')
    ax4.axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_4_pore_paths.png', dpi=300, bbox_inches='tight')
    print(f"已保存: {save_prefix}_4_pore_paths.png")
    plt.close()

    # 图5: 颗粒直径分布
    fig5, ax5 = plt.subplots(figsize=(10, 7))
    ax5.hist(results['diameters'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax5.set_title('Particle Diameter Distribution', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Diameter (pixels)', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.grid(True, alpha=0.3)

    # 添加统计信息
    mean_d = np.mean(results['diameters'])
    std_d = np.std(results['diameters'])
    ax5.axvline(mean_d, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_d:.2f}')
    ax5.axvline(mean_d + std_d, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'Std: ±{std_d:.2f}')
    ax5.axvline(mean_d - std_d, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax5.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_5_diameter_distribution.png', dpi=300, bbox_inches='tight')
    print(f"已保存: {save_prefix}_5_diameter_distribution.png")
    plt.close()

    # 图6: 多边形边数分布
    fig6, ax6 = plt.subplots(figsize=(10, 7))
    sides_unique, counts = np.unique(results['polygon_sides'], return_counts=True)
    ax6.bar(sides_unique, counts, edgecolor='black', alpha=0.7, color='coral')
    ax6.set_title('Polygon Sides Distribution', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Number of Sides', fontsize=12)
    ax6.set_ylabel('Frequency', fontsize=12)
    ax6.grid(True, alpha=0.3, axis='y')

    # 添加统计信息
    most_common_sides = sides_unique[counts.argmax()]
    ax6.axvline(most_common_sides, color='red', linestyle='--', linewidth=2,
                label=f'Most common: {most_common_sides} sides')
    ax6.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_6_polygon_distribution.png', dpi=300, bbox_inches='tight')
    print(f"已保存: {save_prefix}_6_polygon_distribution.png")
    plt.close()

    print(f"\n所有图片已保存完成! 共6张图片，前缀: {save_prefix}")

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 显示最大颗粒容量表
    calculate_max_particles_table()

    # 绘制最大颗粒数量与图像尺寸的关系曲线
    print("\n\nGenerating relationship curves...")
    fig_curve, sizes, all_data = plot_max_particles_vs_size()
    plt.savefig('max_particles_vs_size_combined.png', dpi=200, bbox_inches='tight')
    print("组合图已保存: max_particles_vs_size_combined.png")
    plt.show()

    # 分别保存四个独立图表
    print("\n\nGenerating individual figures...")
    plot_individual_figures(sizes, all_data)
    print("\n所有独立图表已保存完成!")

    # 生成示例 - 256x256图像
    print("\n\nGenerating 256x256 porous media sample...")
    img_size = 256
    n_particles = 4000
    porosity = 0.2

    generator = PorousMediaGenerator(img_size)
    results = generator.generate(
        n_particles=n_particles,
        porosity=porosity,
        distribution='lognormal'
    )

    # 打印分析结果
    print_analysis_results(results)

    # 可视化
    fig = visualize_results(results, img_size)
    plt.savefig('porous_media_analysis.png', dpi=150, bbox_inches='tight')
    # 分别保存各个子图
    visualize_and_save_separately(results, img_size, save_prefix='analysis')
    print("\n\nResults saved to 'porous_media_analysis.png'")
    plt.show()

    # 生成不同尺寸的示例
    print("\n\nGenerating multi-scale examples...")
    fig_multi, axes = plt.subplots(2, 3, figsize=(15, 10))
    sizes_test = [64, 128, 256, 512]

    for idx, size in enumerate(sizes_test):
        row = idx // 2
        col = idx % 2

        # 根据尺寸调整颗粒数
        n_part = int(size / 3)
        gen = PorousMediaGenerator(size)
        res = gen.generate(n_particles=n_part, porosity=0.35)

        axes[row, col].imshow(res['image'], cmap='gray')
        axes[row, col].set_title(f'{size}x{size}\n{n_part} particles, φ={res["porosity"]:.3f}')
        axes[row, col].axis('off')

    # 最后两个子图显示统计信息
    axes[1, 2].text(0.1, 0.9, 'Multi-Scale Generation\nSuccessful!',
                    transform=axes[1, 2].transAxes, fontsize=12,
                    verticalalignment='top')
    axes[1, 2].text(0.1, 0.6,
                    f'GAN Architecture:\n- Attention Mechanism\n- Multi-scale Features\n- Convolutional Layers',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    verticalalignment='top')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('multiscale_porous_media.png', dpi=150, bbox_inches='tight')
    print("Multi-scale results saved to 'multiscale_porous_media.png'")
    plt.show()

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)