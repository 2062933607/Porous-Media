import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage import morphology, measure
import torch
import torch.nn as nn
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
    """改进的Voronoi颗粒生成器"""

    def __init__(self, img_size=256):
        self.img_size = img_size

    def calculate_parameters(self, img_size, target_porosity=0.3,
                             n_particles=None, avg_diameter=None, min_diameter=3):
        """
        智能计算生成参数（三选二模式）

        必须提供以下三个参数中的任意两个：
        1. target_porosity: 目标孔隙率 (0-1)
        2. n_particles: 颗粒数量
        3. avg_diameter: 平均颗粒直径

        参数:
        - img_size: 图像尺寸
        - target_porosity: 目标孔隙率 (0-1)
        - n_particles: 颗粒数量（可选）
        - avg_diameter: 平均颗粒直径（可选）
        - min_diameter: 最小颗粒直径（当需要计算avg_diameter时的下限）

        返回: (n_particles, avg_diameter, target_porosity)
        """
        total_area = img_size ** 2

        # 统计提供了多少个参数
        params_provided = sum([
            target_porosity is not None,
            n_particles is not None,
            avg_diameter is not None
        ])

        if params_provided < 2:
            raise ValueError(
                "必须提供以下三个参数中的任意两个：target_porosity, n_particles, avg_diameter"
            )

        # 情况1: 提供了 porosity 和 n_particles，计算 avg_diameter
        if target_porosity is not None and n_particles is not None and avg_diameter is None:
            particle_area = total_area * (1 - target_porosity)
            avg_particle_area = particle_area / n_particles
            avg_diameter = 2 * np.sqrt(avg_particle_area / np.pi)

            if avg_diameter < min_diameter:
                print(f"警告: 计算得到的平均直径 {avg_diameter:.2f}px 小于最小直径 {min_diameter}px")
                print(f"建议: 减少颗粒数量至 {int(particle_area / (np.pi * (min_diameter / 2) ** 2))} 或更少")

        # 情况2: 提供了 porosity 和 avg_diameter，计算 n_particles
        elif target_porosity is not None and avg_diameter is not None and n_particles is None:
            particle_area = total_area * (1 - target_porosity)
            avg_particle_area = np.pi * (avg_diameter / 2) ** 2
            n_particles = int(particle_area / avg_particle_area)

        # 情况3: 提供了 n_particles 和 avg_diameter，计算 target_porosity
        elif n_particles is not None and avg_diameter is not None and target_porosity is None:
            avg_particle_area = np.pi * (avg_diameter / 2) ** 2
            total_particle_area = n_particles * avg_particle_area
            target_porosity = 1 - (total_particle_area / total_area)

            if target_porosity < 0:
                print(f"警告: 颗粒总面积超过图像面积，计算得到负孔隙率")
                print(f"建议: 减少颗粒数量或减小平均直径")
                target_porosity = 0.05  # 设置最小孔隙率
            elif target_porosity > 0.8:
                print(f"警告: 计算得到的孔隙率 {target_porosity:.1%} 过高")

        return n_particles, avg_diameter, target_porosity

    def generate_particle_diameters(self, n_particles, mean_diameter, std_ratio=0.3,
                                    distribution='lognormal'):
        """生成符合地质统计学规律的颗粒直径分布"""
        if distribution == 'lognormal':
            sigma = std_ratio
            mu = np.log(mean_diameter) - 0.5 * sigma ** 2
            diameters = np.random.lognormal(mu, sigma, n_particles)
        elif distribution == 'gamma':
            k = (1 / std_ratio) ** 2
            theta = mean_diameter / k
            diameters = np.random.gamma(k, theta, n_particles)
        elif distribution == 'weibull':
            k = 2.5
            lambda_ = mean_diameter / (np.gamma(1 + 1 / k))
            diameters = lambda_ * np.random.weibull(k, n_particles)
        else:
            diameters = np.random.normal(mean_diameter, mean_diameter * std_ratio, n_particles)

        # 限制直径范围
        diameters = np.clip(diameters, mean_diameter * 0.3, mean_diameter * 3)
        return diameters

    def generate_polygon_sides(self, n_particles, min_sides=3, max_sides=100):
        """生成多边形边数(3-100边形)"""
        weights = 1.0 / np.arange(min_sides, max_sides + 1)
        weights = weights / weights.sum()
        sides = np.random.choice(range(min_sides, max_sides + 1),
                                 size=n_particles, p=weights)
        return sides

    def create_porous_media(self, n_particles=None, target_porosity=0.3,
                            avg_diameter=None, distribution='lognormal',
                            erosion_factor=0.05, min_diameter=3, verbose=True):
        """
        创建多孔材料图像（新方法，支持灵活参数组合）

        参数组合模式（三选二）：
        - 模式1: 指定 target_porosity + n_particles → 自动计算 avg_diameter
        - 模式2: 指定 target_porosity + avg_diameter → 自动计算 n_particles
        - 模式3: 指定 n_particles + avg_diameter → 自动计算 target_porosity

        参数:
        - n_particles: 颗粒数量（可选）
        - target_porosity: 目标孔隙率 (0-1)（可选）
        - avg_diameter: 平均颗粒直径（可选）
        - distribution: 粒径分布类型 ('lognormal', 'gamma', 'weibull', 'normal')
        - erosion_factor: 侵蚀因子，控制颗粒间隙大小 (0.0-0.15)
        - min_diameter: 最小颗粒直径限制
        - verbose: 是否打印详细信息

        返回: (image, particle_masks, diameters, polygon_sides, actual_porosity)
        """
        # 智能计算参数
        n_particles, avg_diameter, target_porosity = self.calculate_parameters(
            self.img_size, target_porosity, n_particles, avg_diameter, min_diameter
        )

        # 补偿侵蚀造成的面积损失
        compensation_factor = 1.0 / (1 - erosion_factor * 2)
        compensated_diameter = avg_diameter * np.sqrt(compensation_factor)

        if verbose:
            print(f"\n生成参数:")
            print(f"  图像尺寸: {self.img_size}x{self.img_size}")
            print(f"  颗粒数量: {n_particles}")
            print(f"  目标孔隙率: {target_porosity:.2%}")
            print(f"  平均直径: {avg_diameter:.2f} 像素")
            print(f"  补偿后直径: {compensated_diameter:.2f} 像素")
            print(f"  侵蚀因子: {erosion_factor}")

        # 生成颗粒直径
        diameters = self.generate_particle_diameters(n_particles, compensated_diameter,
                                                     distribution=distribution)

        # 生成多边形边数
        polygon_sides = self.generate_polygon_sides(n_particles)

        # 生成Voronoi点
        points = np.random.rand(n_particles, 2) * self.img_size

        # 创建图像
        image = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        particle_masks = []

        # 为每个颗粒创建多边形
        for idx, (point, diameter, n_sides) in enumerate(zip(points, diameters, polygon_sides)):
            radius = diameter / 2

            # 创建多边形顶点
            angles = np.linspace(0, 2 * np.pi, n_sides + 1)[:-1]
            angles += np.random.randn(n_sides) * 0.1
            radii = radius * (1 + np.random.randn(n_sides) * 0.1)

            # 创建颗粒掩码
            mask = np.zeros((self.img_size, self.img_size), dtype=bool)
            y, x = np.ogrid[:self.img_size, :self.img_size]
            center_x, center_y = point[0], point[1]
            dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            # 应用侵蚀因子缩小颗粒
            mask = dist_from_center < radius * (1 - erosion_factor)
            particle_masks.append(mask)
            image[mask] = 1

        # 轻微模糊边缘
        image = gaussian_filter(image.astype(float), sigma=0.5)

        # 计算实际孔隙率
        actual_porosity = (image < 0.5).sum() / image.size

        if verbose:
            print(f"  实际孔隙率: {actual_porosity:.2%}")
            print(f"  偏差: {abs(actual_porosity - target_porosity):.2%}")

        # 将原始直径（未补偿）也返回，用于统计分析
        original_diameters = diameters / np.sqrt(compensation_factor)

        return image, particle_masks, original_diameters, polygon_sides, actual_porosity


# ==================== 孔隙分析 ====================
class PoreAnalyzer:
    """孔隙路径分析器"""

    def __init__(self, image):
        self.image = image
        self.pore_mask = image < 0.5

    def find_all_gaps(self):
        """找到所有缝隙并标记"""
        labeled_pores = measure.label(self.pore_mask)
        n_pores = labeled_pores.max()

        gap_colored = np.zeros((*self.image.shape, 3))
        colors = plt.cm.rainbow(np.linspace(0, 1, max(n_pores, 1)))

        gap_info = []
        for i in range(1, n_pores + 1):
            mask = labeled_pores == i
            area = mask.sum()
            gap_colored[mask] = colors[i - 1][:3]
            gap_info.append({'id': i, 'area': area, 'mask': mask})

        return gap_colored, gap_info

    def find_widest_path(self):
        """找到最宽的连通路径"""
        distance = distance_transform_edt(self.pore_mask)
        skeleton = morphology.skeletonize(self.pore_mask)
        widths = distance[skeleton]

        if len(widths) == 0:
            return None, 0

        max_width = widths.max() * 2
        widest_path = np.zeros_like(self.image)
        threshold = max_width * 0.8
        widest_path[distance >= threshold / 2] = 1

        return widest_path, max_width

    def trace_pore_paths(self):
        """追踪孔隙路径"""
        skeleton = morphology.skeletonize(self.pore_mask)

        y_coords, x_coords = np.where(skeleton)

        if len(y_coords) == 0:
            return skeleton

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
            'cv': np.std(diameters) / np.mean(diameters),
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

    def generate(self, n_particles=None, porosity=None, avg_diameter=None,
                 distribution='lognormal', erosion_factor=0.05, min_diameter=3):
        """
        生成多孔材料图像（支持灵活参数组合）

        参数组合（三选二）：
        1. porosity + n_particles → 自动计算 avg_diameter
        2. porosity + avg_diameter → 自动计算 n_particles
        3. n_particles + avg_diameter → 自动计算 porosity

        参数:
        - n_particles: 颗粒数量（可选）
        - porosity: 孔隙率 (0-1)（可选）
        - avg_diameter: 平均颗粒直径（可选）
        - distribution: 粒径分布类型
        - erosion_factor: 侵蚀因子 (0.0-0.15)
        - min_diameter: 最小颗粒直径

        返回: 包含所有分析结果的字典
        """
        # 生成基础图像
        image, masks, diameters, polygon_sides, actual_porosity = \
            self.voronoi_gen.create_porous_media(
                n_particles=n_particles,
                target_porosity=porosity,
                avg_diameter=avg_diameter,
                distribution=distribution,
                erosion_factor=erosion_factor,
                min_diameter=min_diameter
            )

        # 孔隙分析
        analyzer = PoreAnalyzer(image)
        gap_colored, gap_info = analyzer.find_all_gaps()
        widest_path, max_width = analyzer.find_widest_path()
        pore_paths = analyzer.trace_pore_paths()

        # 地质统计分析
        geo_analyzer = GeostatisticalAnalyzer(image)
        particle_stats = geo_analyzer.particle_size_distribution(diameters)
        percolates, perc_ratio = geo_analyzer.percolation_analysis()
        hydraulic_k = geo_analyzer.hydraulic_conductivity_estimate(diameters)

        results = {
            'image': image,
            'gap_colored': gap_colored,
            'widest_path': widest_path,
            'pore_paths': pore_paths,
            'max_width': max_width,
            'porosity': actual_porosity,
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


def show_parameter_combinations():
    """显示参数组合示例"""
    print("\n" + "=" * 100)
    print("参数组合使用指南")
    print("=" * 100)

    print("\n【模式1】指定 孔隙率 + 颗粒数 → 自动计算直径")
    print("-" * 100)
    print("示例: generator.generate(porosity=0.25, n_particles=2000)")
    print("适用: 已知材料特性和颗粒密度，需要确定颗粒大小")

    print("\n【模式2】指定 孔隙率 + 直径 → 自动计算颗粒数")
    print("-" * 100)
    print("示例: generator.generate(porosity=0.25, avg_diameter=6)")
    print("适用: 已知材料特性和颗粒尺寸，需要确定颗粒数量")
    print("推荐: 最常用的模式，直观易懂")

    print("\n【模式3】指定 颗粒数 + 直径 → 自动计算孔隙率")
    print("-" * 100)
    print("示例: generator.generate(n_particles=2000, avg_diameter=6)")
    print("适用: 已知颗粒配置，需要预测孔隙率")

    print("\n" + "=" * 100)
    print("推荐参数范围:")
    print("  - 孔隙率: 0.15-0.45 (岩石 0.15-0.25, 砂土 0.25-0.35, 疏松材料 0.35-0.45)")
    print("  - 颗粒数: 500-5000 (256x256图像)")
    print("  - 平均直径: 4-10 像素")
    print("  - 侵蚀因子: 0.02-0.08 (越小越接近目标孔隙率)")
    print("=" * 100)


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 显示参数组合使用指南
    show_parameter_combinations()

    # 示例1: 孔隙率 + 平均直径 → 自动计算颗粒数
    print("\n\n" + "=" * 80)
    print("【示例1】模式2: 指定孔隙率 + 平均直径")
    print("=" * 80)

    img_size = 256
    generator = PorousMediaGenerator(img_size)

    results1 = generator.generate(
        porosity=0.20,  # 目标孔隙率20%
        avg_diameter=6,  # 平均直径6像素
        erosion_factor=0.02  # 较小的侵蚀因子
    )

    print_analysis_results(results1)
    fig1 = visualize_results(results1, img_size)
    plt.savefig('example1_porosity_diameter.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存: example1_porosity_diameter.png")
    plt.show()

    # 示例2: 孔隙率 + 颗粒数 → 自动计算直径
    print("\n\n" + "=" * 80)
    print("【示例2】模式1: 指定孔隙率 + 颗粒数")
    print("=" * 80)

    results2 = generator.generate(
        porosity=0.3,  # 目标孔隙率30%
        n_particles=4012,  # 2000个颗粒
        erosion_factor=0.005
    )

    print_analysis_results(results2)
    fig2 = visualize_results(results2, img_size)
    plt.savefig('example2_porosity_particles.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存: example2_porosity_particles.png")
    plt.show()

    # 示例3: 颗粒数 + 平均直径 → 自动计算孔隙率
    print("\n\n" + "=" * 80)
    print("【示例3】模式3: 指定颗粒数 + 平均直径")
    print("=" * 80)

    results3 = generator.generate(
        n_particles=3000,  # 1500个颗粒
        avg_diameter=4,  # 平均直径7像素
        erosion_factor=0.002
    )

    print_analysis_results(results3)
    fig3 = visualize_results(results3, img_size)
    plt.savefig('example3_particles_diameter.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存: example3_particles_diameter.png")
    plt.show()

    # 对比不同孔隙率
    print("\n\n" + "=" * 80)
    print("【对比测试】不同孔隙率对比 (固定直径=6px)")
    print("=" * 80)

    fig_compare, axes = plt.subplots(2, 3, figsize=(15, 10))
    porosities = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    for idx, target_por in enumerate(porosities):
        row = idx // 3
        col = idx % 3

        res = generator.generate(
            porosity=target_por,
            avg_diameter=6,
            erosion_factor=0.005
        )

        axes[row, col].imshow(res['image'], cmap='gray')
        axes[row, col].set_title(
            f'Target: {target_por:.0%}\n'
            f'Actual: {res["porosity"]:.1%}\n'
            f'Particles: {len(res["diameters"])}'
        )
        axes[row, col].axis('off')

    plt.suptitle('Porosity Comparison (avg_diameter=6px, erosion=0.03)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('porosity_comparison.png', dpi=150, bbox_inches='tight')
    print("\n对比结果已保存: porosity_comparison.png")
    plt.show()

    # 对比不同颗粒数
    print("\n\n" + "=" * 80)
    print("【对比测试】不同颗粒数对比 (固定孔隙率=25%)")
    print("=" * 80)

    fig_particles, axes = plt.subplots(2, 3, figsize=(15, 10))
    particle_counts = [500, 1000, 1500, 2000, 3000, 4000]

    for idx, n_part in enumerate(particle_counts):
        row = idx // 3
        col = idx % 3

        res = generator.generate(
            porosity=0.35,
            n_particles=n_part,
            erosion_factor=0.005
        )

        avg_d = np.mean(res['diameters'])

        axes[row, col].imshow(res['image'], cmap='gray')
        axes[row, col].set_title(
            f'Particles: {n_part}\n'
            f'Avg D: {avg_d:.1f}px\n'
            f'Porosity: {res["porosity"]:.1%}'
        )
        axes[row, col].axis('off')

    plt.suptitle('Particle Count Comparison (porosity=25%, erosion=0.03)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('particle_count_comparison.png', dpi=150, bbox_inches='tight')
    print("\n对比结果已保存: particle_count_comparison.png")
    plt.show()

    # 对比不同直径
    print("\n\n" + "=" * 80)
    print("【对比测试】不同平均直径对比 (固定孔隙率=25%)")
    print("=" * 80)

    fig_diameter, axes = plt.subplots(2, 3, figsize=(15, 10))
    diameters = [4, 5, 6, 7, 8, 10]

    for idx, d in enumerate(diameters):
        row = idx // 3
        col = idx % 3

        res = generator.generate(
            porosity=0.25,
            avg_diameter=d,
            erosion_factor=0.005
        )

        axes[row, col].imshow(res['image'], cmap='gray')
        axes[row, col].set_title(
            f'Avg D: {d}px\n'
            f'Particles: {len(res["diameters"])}\n'
            f'Porosity: {res["porosity"]:.1%}'
        )
        axes[row, col].axis('off')

    plt.suptitle('Diameter Comparison (porosity=25%, erosion=0.03)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('diameter_comparison.png', dpi=150, bbox_inches='tight')
    print("\n对比结果已保存: diameter_comparison.png")
    plt.show()

    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print("\n✓ 所有旧方法已移除，现在使用统一的新方法")
    print("✓ 支持三种灵活的参数组合模式（三选二）")
    print("✓ 自动计算缺失参数，使用更加便捷")
    print("\n推荐使用:")
    print("  generator.generate(porosity=0.25, avg_diameter=6)  # 最直观")
    print("  generator.generate(porosity=0.25, n_particles=2000)")
    print("  generator.generate(n_particles=2000, avg_diameter=6)")