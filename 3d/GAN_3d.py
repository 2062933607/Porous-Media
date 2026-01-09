import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Voronoi
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_erosion
from skimage import morphology, measure
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')


# ==================== 3D GAN架构 ====================
class AttentionBlock3D(nn.Module):
    """3D注意力机制模块"""

    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv3d(channels, channels // 8, 1)
        self.key = nn.Conv3d(channels, channels // 8, 1)
        self.value = nn.Conv3d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, D, H, W = x.size()
        q = self.query(x).view(B, -1, D * H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, D * H * W)
        v = self.value(x).view(B, -1, D * H * W)

        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(B, C, D, H, W)

        return self.gamma * out + x


class Generator3D(nn.Module):
    """3D生成器"""

    def __init__(self, latent_dim=100, vol_size=64):
        super().__init__()
        self.init_size = vol_size // 8
        self.l1 = nn.Linear(latent_dim, 512 * self.init_size ** 3)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm3d(512),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(512, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2),
            nn.Conv3d(256, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            AttentionBlock3D(128),

            nn.Upsample(scale_factor=2),
            nn.Conv3d(128, 1, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.size(0), 512, self.init_size, self.init_size, self.init_size)
        vol = self.conv_blocks(out)
        return vol


class Discriminator3D(nn.Module):
    """3D判别器"""

    def __init__(self, vol_size=64):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv3d(1, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout3d(0.25),

            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout3d(0.25),

            nn.Conv3d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout3d(0.25),
        )

        ds_size = vol_size // 8
        self.adv_layer = nn.Sequential(
            nn.Linear(256 * ds_size ** 3, 1),
            nn.Sigmoid()
        )

    def forward(self, vol):
        out = self.model(vol)
        out = out.view(out.size(0), -1)
        validity = self.adv_layer(out)
        return validity


# ==================== 3D Voronoi颗粒生成 ====================
class Voronoi3DParticleGenerator:
    """3D Voronoi颗粒生成器"""

    def __init__(self, vol_size=64):
        self.vol_size = vol_size

    def calculate_max_particles(self, vol_size, min_diameter=3, packing_ratio=0.65):
        """
        计算3D最大允许颗粒数量

        参数:
        - vol_size: 体素尺寸
        - min_diameter: 最小颗粒直径(体素)
        - packing_ratio: 有效堆积率(0-1)
          * 0.60-0.65: 随机堆积球体(理论最大~64%)
          * 0.50-0.60: 松散堆积
          * 0.65-0.74: 密实堆积(理论最大FCC~74%)

        返回: 最大颗粒数量
        """
        volume = vol_size ** 3
        min_particle_volume = (4 / 3) * np.pi * (min_diameter / 2) ** 3
        max_particles = int(volume * packing_ratio / min_particle_volume)
        return max_particles

    def generate_particle_diameters(self, n_particles, mean_diameter, std_ratio=0.3,
                                    distribution='lognormal'):
        """生成符合地质统计学规律的3D颗粒直径分布"""
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

        diameters = np.clip(diameters, mean_diameter * 0.3, mean_diameter * 3)
        return diameters

    def create_3d_polyhedron(self, center, diameter, n_faces=None):
        """创建3D多面体(4面体到100面体近似球体)"""
        if n_faces is None:
            # 随机选择面数: 4(四面体), 6(立方体), 8(八面体), 12(十二面体), 20(二十面体)
            # 或更多面近似球体
            face_options = [4, 6, 8, 12, 20] + list(range(30, 101, 10))
            weights = np.array([1.0 / len(face_options)] * len(face_options))
            n_faces = np.random.choice(face_options, p=weights)

        radius = diameter / 2

        # 生成近似均匀分布的点(Fibonacci球面)
        n_vertices = max(n_faces, 20)
        indices = np.arange(0, n_vertices, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n_vertices)
        theta = np.pi * (1 + 5 ** 0.5) * indices

        # 添加随机扰动使其不规则
        perturbation = 0.15
        phi += np.random.randn(n_vertices) * perturbation
        theta += np.random.randn(n_vertices) * perturbation

        # 半径随机变化
        radii = radius * (1 + np.random.randn(n_vertices) * 0.1)

        x = radii * np.sin(phi) * np.cos(theta) + center[0]
        y = radii * np.sin(phi) * np.sin(theta) + center[1]
        z = radii * np.cos(phi) + center[2]

        return np.column_stack([x, y, z])

    def create_voronoi_with_gaps_3d(self, n_particles, porosity=0.35,
                                    mean_diameter=None, std_ratio=0.3, distribution='lognormal'):
        """创建带间隙的3D Voronoi颗粒体

        参数:
        - n_particles: 颗粒数量
        - porosity: 目标孔隙率 (0-1)
        - mean_diameter: 平均直径(体素), 如果为None则自动计算
        - std_ratio: 直径标准差比例 (默认0.3表示CV=30%)
        - distribution: 分布类型 ('lognormal', 'gamma', 'weibull', 'normal')
        """
        if mean_diameter is None:
            effective_volume = self.vol_size ** 3 * (1 - porosity)
            mean_diameter = ((6 * effective_volume / n_particles / np.pi) ** (1 / 3)) * 2
            print(f"自动计算平均直径: {mean_diameter:.2f} voxels")

        # 生成颗粒直径
        diameters = self.generate_particle_diameters(n_particles, mean_diameter, std_ratio,
                                                     distribution=distribution)

        # 生成Voronoi点
        points = np.random.rand(n_particles, 3) * self.vol_size

        # 创建3D体素图像
        volume = np.zeros((self.vol_size, self.vol_size, self.vol_size), dtype=np.float32)

        # 为每个颗粒创建多面体
        for idx, (point, diameter) in enumerate(zip(points, diameters)):
            radius = diameter / 2

            # 创建球形颗粒掩码
            z, y, x = np.ogrid[:self.vol_size, :self.vol_size, :self.vol_size]
            dist_from_center = np.sqrt((x - point[0]) ** 2 +
                                       (y - point[1]) ** 2 +
                                       (z - point[2]) ** 2)

            # 稍微缩小以产生间隙
            mask = dist_from_center < radius * 0.92
            volume[mask] = 1

        # 应用3D腐蚀以创建间隙
        gap_size = max(1, int(mean_diameter * 0.05))
        struct_elem = morphology.ball(gap_size)
        volume = binary_erosion(volume, struct_elem)

        # 模糊边缘
        volume = gaussian_filter(volume.astype(float), sigma=1.0)

        return volume, diameters, points


# ==================== 3D孔隙分析 ====================
class Pore3DAnalyzer:
    """3D孔隙路径分析器"""

    def __init__(self, volume):
        self.volume = volume
        self.pore_mask = volume < 0.5

    def find_all_gaps_3d(self):
        """找到所有3D缝隙并标记"""
        labeled_pores = measure.label(self.pore_mask)
        n_pores = labeled_pores.max()

        gap_info = []
        for i in range(1, n_pores + 1):
            mask = labeled_pores == i
            volume_size = mask.sum()
            gap_info.append({
                'id': i,
                'volume': volume_size,
                'mask': mask
            })

        return labeled_pores, gap_info

    def find_widest_path_3d(self):
        """找到3D最宽的连通路径"""
        distance = distance_transform_edt(self.pore_mask)

        if distance.max() == 0:
            return None, 0

        max_width = distance.max() * 2

        widest_path = np.zeros_like(self.volume)
        threshold = max_width * 0.8
        widest_path[distance >= threshold / 2] = 1

        return widest_path, max_width

    def percolation_analysis_3d(self):
        """3D渗透性分析(检查z方向连通性)"""
        labeled = measure.label(self.pore_mask)

        top_labels = set(labeled[0, :, :].flatten())
        bottom_labels = set(labeled[-1, :, :].flatten())
        percolating_labels = top_labels & bottom_labels
        percolating_labels.discard(0)

        percolates = len(percolating_labels) > 0

        if percolates:
            largest_cluster = max(percolating_labels,
                                  key=lambda x: (labeled == x).sum())
            percolation_ratio = (labeled == largest_cluster).sum() / self.pore_mask.sum()
        else:
            percolation_ratio = 0

        return percolates, percolation_ratio


# ==================== 3D地质统计分析 ====================
class Geostatistical3DAnalyzer:
    """3D地质统计学分析器"""

    def __init__(self, volume):
        self.volume = volume
        self.porosity = (volume < 0.5).sum() / volume.size

    def calculate_porosity(self):
        """计算3D孔隙率"""
        return self.porosity

    def particle_size_distribution_3d(self, diameters):
        """3D颗粒粒径分布分析"""
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

    def hydraulic_conductivity_estimate_3d(self, diameters):
        """3D水力传导度估算(Kozeny-Carman方程)"""
        porosity = self.porosity
        d50 = np.percentile(diameters, 50)

        # 3D Kozeny-Carman方程
        k = (d50 ** 2 * porosity ** 3) / (180 * (1 - porosity) ** 2)

        return k


# ==================== 3D主系统 ====================
class PorousMedia3DGenerator:
    """3D多孔材料生成主系统"""

    def __init__(self, vol_size=64):
        self.vol_size = vol_size
        self.voronoi_gen = Voronoi3DParticleGenerator(vol_size)

    def generate(self, n_particles, porosity=0.35, mean_diameter=None, std_ratio=0.3,
                 distribution='lognormal'):
        """生成3D多孔材料

        参数:
        - n_particles: 颗粒数量
        - porosity: 目标孔隙率 (0-1)
        - mean_diameter: 平均直径(体素), None表示自动计算
        - std_ratio: 直径标准差比例 (默认0.3)
        - distribution: 分布类型 ('lognormal', 'gamma', 'weibull', 'normal')
        """
        # 生成基础3D Voronoi体
        volume, diameters, points = self.voronoi_gen.create_voronoi_with_gaps_3d(
            n_particles, porosity, mean_diameter, std_ratio, distribution
        )

        # 3D孔隙分析
        analyzer = Pore3DAnalyzer(volume)
        labeled_pores, gap_info = analyzer.find_all_gaps_3d()
        widest_path, max_width = analyzer.find_widest_path_3d()
        percolates, perc_ratio = analyzer.percolation_analysis_3d()

        # 3D地质统计分析
        geo_analyzer = Geostatistical3DAnalyzer(volume)
        porosity_actual = geo_analyzer.calculate_porosity()
        particle_stats = geo_analyzer.particle_size_distribution_3d(diameters)
        hydraulic_k = geo_analyzer.hydraulic_conductivity_estimate_3d(diameters)

        results = {
            'volume': volume,
            'labeled_pores': labeled_pores,
            'widest_path': widest_path,
            'max_width': max_width,
            'porosity': porosity_actual,
            'n_gaps': len(gap_info),
            'particle_stats': particle_stats,
            'diameters': diameters,
            'points': points,
            'percolates': percolates,
            'percolation_ratio': perc_ratio,
            'hydraulic_conductivity': hydraulic_k
        }

        return results


# ==================== 分别保存六张独立图片 ====================
def save_individual_3d_views(results, vol_size):
    """分别保存六个独立的3D视图为单独的图片文件"""
    mid_slice = results['volume'][vol_size // 2, :, :]

    # 视图1: 中间切片
    print("\n正在保存视图1: 中间切片...")
    fig1, ax1 = plt.subplots(figsize=(10, 9))
    im1 = ax1.imshow(mid_slice, cmap='gray')
    ax1.set_title(f'Middle Slice (Z={vol_size // 2})\nPorosity: {results["porosity"]:.3f}',
                  fontsize=16, fontweight='bold', pad=15)
    ax1.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Solid Fraction', fontsize=12)
    plt.tight_layout()
    plt.savefig('3d_view1_middle_slice.png', dpi=300, bbox_inches='tight')
    print("✓ 3D视图1已保存: 3d_view1_middle_slice.png")
    plt.close()

    # 视图2: 3D体渲染
    print("正在保存视图2: 3D颗粒结构...")
    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(111, projection='3d')
    vol = results['volume']
    stride = max(1, vol_size // 32)
    x, y, z = np.where(vol[::stride, ::stride, ::stride] > 0.5)
    x, y, z = x * stride, y * stride, z * stride
    scatter = ax2.scatter(x, y, z, c=vol[x, y, z], cmap='viridis', s=2, alpha=0.6)
    ax2.set_xlabel('X (voxels)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Y (voxels)', fontsize=12, fontweight='bold')
    ax2.set_zlabel('Z (voxels)', fontsize=12, fontweight='bold')
    ax2.set_title(f'3D Particle Structure\n{len(results["points"])} particles',
                  fontsize=16, fontweight='bold', pad=20)
    cbar2 = plt.colorbar(scatter, ax=ax2, fraction=0.03, pad=0.1)
    cbar2.set_label('Solid Fraction', fontsize=12)
    plt.tight_layout()
    plt.savefig('3d_view2_particle_structure.png', dpi=300, bbox_inches='tight')
    print("✓ 3D视图2已保存: 3d_view2_particle_structure.png")
    plt.close()

    # 视图3: 孔隙网络切片
    print("正在保存视图3: 孔隙网络...")
    fig3, ax3 = plt.subplots(figsize=(10, 9))
    pore_slice = results['labeled_pores'][vol_size // 2, :, :]
    im3 = ax3.imshow(pore_slice, cmap='nipy_spectral', interpolation='nearest')
    ax3.set_title(f'Pore Network (Z={vol_size // 2})\n{results["n_gaps"]} interconnected pores',
                  fontsize=16, fontweight='bold', pad=15)
    ax3.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Pore ID', fontsize=12)
    plt.tight_layout()
    plt.savefig('3d_view3_pore_network.png', dpi=300, bbox_inches='tight')
    print("✓ 3D视图3已保存: 3d_view3_pore_network.png")
    plt.close()

    # 视图4: 最宽路径
    print("正在保存视图4: 最宽渗透路径...")
    fig4, ax4 = plt.subplots(figsize=(10, 9))
    ax4.imshow(mid_slice, cmap='gray', alpha=0.7)
    if results['widest_path'] is not None:
        path_slice = results['widest_path'][vol_size // 2, :, :]
        im4 = ax4.imshow(path_slice, cmap='Reds', alpha=0.7)
        ax4.set_title(f'Widest Percolation Path\nMax width: {results["max_width"]:.2f} voxels',
                      fontsize=16, fontweight='bold', pad=15)
    else:
        ax4.set_title('No Percolating Path Found', fontsize=16, fontweight='bold', pad=15)
    ax4.axis('off')
    plt.tight_layout()
    plt.savefig('3d_view4_widest_path.png', dpi=300, bbox_inches='tight')
    print("✓ 3D视图4已保存: 3d_view4_widest_path.png")
    plt.close()

    # 视图5: 直径分布
    print("正在保存视图5: 颗粒直径分布...")
    fig5, ax5 = plt.subplots(figsize=(10, 7))
    n, bins, patches = ax5.hist(results['diameters'], bins=30, edgecolor='black',
                                alpha=0.7, color='skyblue', linewidth=1.5)

    # 添加统计线
    mean_d = results['particle_stats']['mean']
    median_d = results['particle_stats']['d50']
    ax5.axvline(mean_d, color='red', linestyle='--', linewidth=2.5,
                label=f'Mean: {mean_d:.2f}')
    ax5.axvline(median_d, color='green', linestyle='--', linewidth=2.5,
                label=f'Median: {median_d:.2f}')

    ax5.set_xlabel('Diameter (voxels)', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax5.set_title('Particle Diameter Distribution\n(Lognormal Distribution)',
                  fontsize=16, fontweight='bold', pad=15)
    ax5.legend(fontsize=12, loc='upper right')
    ax5.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('3d_view5_diameter_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ 3D视图5已保存: 3d_view5_diameter_distribution.png")
    plt.close()

    # 视图6: 统计信息
    print("正在保存视图6: 统计分析...")
    fig6, ax6 = plt.subplots(figsize=(10, 10))
    ax6.axis('off')

    stats_text = f"""
    3D POROUS MEDIA STATISTICS
    {'=' * 50}

    GEOMETRY
    {'─' * 50}
    Volume Size:              {vol_size}³ voxels
    Total Voxels:             {vol_size ** 3:,}
    Number of Particles:      {len(results['points'])}

    POROSITY ANALYSIS
    {'─' * 50}
    Porosity:                 {results['porosity']:.4f} ({results['porosity'] * 100:.2f}%)
    Solid Fraction:           {1 - results['porosity']:.4f} ({(1 - results['porosity']) * 100:.2f}%)
    Number of Pores:          {results['n_gaps']}
    Max Pore Width:           {results['max_width']:.2f} voxels

    CONNECTIVITY
    {'─' * 50}
    Percolation (z-dir):      {'Yes ✓' if results['percolates'] else 'No ✗'}
    Percolation Ratio:        {results['percolation_ratio']:.4f}
    Hydraulic Conductivity:   {results['hydraulic_conductivity']:.2e}

    PARTICLE SIZE STATISTICS
    {'─' * 50}
    Mean Diameter:            {results['particle_stats']['mean']:.2f} voxels
    Std Deviation:            {results['particle_stats']['std']:.2f} voxels
    Coefficient of Variation: {results['particle_stats']['cv']:.4f}
    d10 (10th percentile):    {results['particle_stats']['d10']:.2f} voxels
    d50 (median):             {results['particle_stats']['d50']:.2f} voxels
    d90 (90th percentile):    {results['particle_stats']['d90']:.2f} voxels
    Uniformity Coefficient:   {results['particle_stats']['uniformity']:.4f}
    Curvature Coefficient:    {results['particle_stats']['curvature']:.4f}

    {'=' * 50}
    Generated using 3D GAN-Voronoi Algorithm
    Distribution: Lognormal
    """

    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, pad=15))

    ax6.set_title('Complete Statistical Analysis',
                  fontsize=18, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('3d_view6_statistics.png', dpi=300, bbox_inches='tight')
    print("✓ 3D视图6已保存: 3d_view6_statistics.png")
    plt.close()

    print("\n" + "=" * 60)
    print("所有6张独立视图已成功保存!")
    print("=" * 60)


# ==================== 3D容量计算和曲线 ====================
def calculate_max_particles_3d_table():
    """计算3D不同体素尺寸的最大颗粒数量"""
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    print("\n" + "=" * 90)
    print("3D MAXIMUM PARTICLE CAPACITY (不同堆积率对比)")
    print("=" * 90)

    packing_ratios = [0.50, 0.60, 0.65, 0.74]
    packing_names = ['松散(φ=50%)', '随机(φ=40%)', '中等(φ=35%)', '密实(φ=26%)']

    for ratio, name in zip(packing_ratios, packing_names):
        print(f"\n{name} - Packing Ratio = {ratio:.2f}")
        print("-" * 90)
        print(f"{'体素尺寸':<18} {'最大颗粒数':<18} {'单颗粒体积':<22} {'实际孔隙率':<18}")
        print("-" * 90)

        for size in sizes:
            gen = Voronoi3DParticleGenerator(size)
            max_particles = gen.calculate_max_particles(size, min_diameter=3, packing_ratio=ratio)
            volume_per_particle = (size ** 3) / max_particles
            actual_porosity = 1 - ratio
            print(f"{size}×{size}×{size:<10} {max_particles:<18,} {volume_per_particle:<22.2f} {actual_porosity:.1%}")

    print("\n" + "=" * 90)
    print("说明:")
    print("  - 基于最小颗粒直径3体素")
    print("  - 3D随机球体堆积理论极限: ~64%")
    print("  - 3D面心立方(FCC)最密堆积: ~74%")
    print("  - 天然3D多孔介质孔隙率: 25-45%")


def plot_max_particles_3d_vs_size():
    """绘制3D最大颗粒数量与体素尺寸的关系曲线(组合图)"""
    sizes = np.linspace(16, 512, 80)  # 3D计算量大,限制到512

    packing_ratios = [0.50, 0.60, 0.65, 0.74]
    packing_names = ['松散 (φ=50%)', '随机 (φ=40%)', '中等 (φ=35%)', '密实FCC (φ=26%)']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    markers = ['o', 's', '^', 'D']

    # 预计算所有数据
    all_data = {}
    for ratio in packing_ratios:
        max_particles_list = []
        density_list = []
        for size in sizes:
            gen = Voronoi3DParticleGenerator(int(size))
            max_p = gen.calculate_max_particles(int(size), min_diameter=3, packing_ratio=ratio)
            max_particles_list.append(max_p)
            density = max_p / (size ** 3)
            density_list.append(density * 10000)  # 颗粒/10000体素³
        all_data[ratio] = {'particles': max_particles_list, 'density': density_list}

    # 创建组合图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 子图1: 线性坐标
    ax1 = axes[0, 0]
    for ratio, name, color, marker in zip(packing_ratios, packing_names, colors, markers):
        max_particles_list = all_data[ratio]['particles']
        ax1.plot(sizes, max_particles_list, label=name, color=color, linewidth=2.5)
        key_sizes = [16, 64, 128, 256, 512]
        key_indices = [np.argmin(np.abs(sizes - s)) for s in key_sizes]
        ax1.scatter([sizes[i] for i in key_indices],
                    [max_particles_list[i] for i in key_indices],
                    color=color, s=80, marker=marker, zorder=5)

    ax1.set_xlabel('体素尺寸 (voxels)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('最大颗粒数量', fontsize=12, fontweight='bold')
    ax1.set_title('3D最大颗粒数量 vs 体素尺寸 (线性坐标)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([0, 520])

    # 子图2: 对数坐标
    ax2 = axes[0, 1]
    for ratio, name, color, marker in zip(packing_ratios, packing_names, colors, markers):
        max_particles_list = all_data[ratio]['particles']
        ax2.loglog(sizes, max_particles_list, label=name, color=color, linewidth=2.5)
        key_sizes = [16, 64, 256]
        key_indices = [np.argmin(np.abs(sizes - s)) for s in key_sizes]
        ax2.scatter([sizes[i] for i in key_indices],
                    [max_particles_list[i] for i in key_indices],
                    color=color, s=80, marker=marker, zorder=5)

    # 添加理论曲线 N ∝ size³
    theoretical = (sizes / sizes[0]) ** 3 * 65
    ax2.plot(sizes, theoretical, 'k--', linewidth=2, alpha=0.5, label='理论: N ∝ size³')

    ax2.set_xlabel('体素尺寸 (voxels)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('最大颗粒数量', fontsize=12, fontweight='bold')
    ax2.set_title('3D最大颗粒数量 vs 体素尺寸 (对数坐标)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--', which='both')

    # 子图3: 颗粒密度
    ax3 = axes[1, 0]
    for ratio, name, color, marker in zip(packing_ratios, packing_names, colors, markers):
        density_list = all_data[ratio]['density']
        ax3.plot(sizes, density_list, label=name, color=color, linewidth=2.5)
        key_sizes = [16, 128, 256, 512]
        key_indices = [np.argmin(np.abs(sizes - s)) for s in key_sizes]
        ax3.scatter([sizes[i] for i in key_indices],
                    [density_list[i] for i in key_indices],
                    color=color, s=80, marker=marker, zorder=5)

    ax3.set_xlabel('体素尺寸 (voxels)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('颗粒密度 (颗粒/10000体素³)', fontsize=12, fontweight='bold')
    ax3.set_title('3D颗粒密度 vs 体素尺寸', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim([0, 520])

    # 子图4: 数据表格
    ax4 = axes[1, 1]
    ax4.axis('off')

    table_data = []
    table_data.append(['体素尺寸', '松散\n(φ=50%)', '随机\n(φ=40%)', '中等\n(φ=35%)', '密实FCC\n(φ=26%)'])

    key_sizes_table = [16, 32, 64, 128, 256, 512]
    for size in key_sizes_table:
        row = [f'{size}³']
        for ratio in packing_ratios:
            gen = Voronoi3DParticleGenerator(size)
            max_p = gen.calculate_max_particles(size, min_diameter=3, packing_ratio=ratio)
            if max_p >= 1000000:
                row.append(f'{max_p / 1e6:.2f}M')
            elif max_p >= 1000:
                row.append(f'{max_p / 1e3:.1f}K')
            else:
                row.append(f'{max_p}')
        table_data.append(row)

    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.15, 0.18, 0.18, 0.18, 0.18])
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

    ax4.set_title('3D最大颗粒数量对照表', fontsize=14, fontweight='bold', pad=20)

    # 添加总标题和说明
    fig.suptitle('3D最大允许颗粒数量 vs 体素尺寸关系分析',
                 fontsize=16, fontweight='bold', y=0.995)

    fig.text(0.5, 0.02,
             '说明: φ=孔隙率, 基于最小颗粒直径3体素 | 理论关系: 最大颗粒数 ∝ (体素尺寸)³ | FCC=面心立方最密堆积',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0.03, 1, 0.99])

    return fig, sizes, all_data


def plot_individual_3d_figures(sizes, all_data):
    """分别绘制四个独立的3D图表"""
    packing_ratios = [0.50, 0.60, 0.65, 0.74]
    packing_names = ['φ=50%', 'φ=40%', 'φ=35%', 'φ=26%']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    markers = ['o', 's', '^', 'D']

    # 图1: 线性坐标
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    for ratio, name, color, marker in zip(packing_ratios, packing_names, colors, markers):
        max_particles_list = all_data[ratio]['particles']
        ax1.plot(sizes, max_particles_list, label=name, color=color, linewidth=3)
        key_sizes = [16, 64, 128, 256, 512]
        key_indices = [np.argmin(np.abs(sizes - s)) for s in key_sizes]
        ax1.scatter([sizes[i] for i in key_indices],
                    [max_particles_list[i] for i in key_indices],
                    color=color, s=100, marker=marker, zorder=5, edgecolors='black', linewidths=1.5)

    ax1.set_xlabel('Voxel size', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Maximum number of particles', fontsize=14, fontweight='bold')
    ax1.set_title('Maximum number of particles under different porosities in 3D', fontsize=16, fontweight='bold', pad=15)
    ax1.legend(fontsize=12, loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([0, 520])
    plt.tight_layout()
    plt.savefig('3d_figure1_linear_scale.png', dpi=300, bbox_inches='tight')
    print("3D图1已保存: 3d_figure1_linear_scale.png")
    plt.close()

    # 图2: 对数坐标
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    for ratio, name, color, marker in zip(packing_ratios, packing_names, colors, markers):
        max_particles_list = all_data[ratio]['particles']
        ax2.loglog(sizes, max_particles_list, label=name, color=color, linewidth=3)
        key_sizes = [16, 64, 256]
        key_indices = [np.argmin(np.abs(sizes - s)) for s in key_sizes]
        ax2.scatter([sizes[i] for i in key_indices],
                    [max_particles_list[i] for i in key_indices],
                    color=color, s=100, marker=marker, zorder=5, edgecolors='black', linewidths=1.5)

    # 添加理论曲线
    theoretical = (sizes / sizes[0]) ** 3 * 65
    ax2.plot(sizes, theoretical, 'k--', linewidth=2.5, alpha=0.6, label='理论: N ∝ size³')

    ax2.set_xlabel('体素尺寸 (voxels)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('最大颗粒数量', fontsize=14, fontweight='bold')
    ax2.set_title('3D最大颗粒数量 vs 体素尺寸 (对数坐标)', fontsize=16, fontweight='bold', pad=15)
    ax2.legend(fontsize=12, loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', which='both')
    plt.tight_layout()
    plt.savefig('3d_figure2_log_scale.png', dpi=300, bbox_inches='tight')
    print("3D图2已保存: 3d_figure2_log_scale.png")
    plt.close()

    # 图3: 颗粒密度
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    for ratio, name, color, marker in zip(packing_ratios, packing_names, colors, markers):
        density_list = all_data[ratio]['density']
        ax3.plot(sizes, density_list, label=name, color=color, linewidth=3)
        key_sizes = [16, 128, 256, 512]
        key_indices = [np.argmin(np.abs(sizes - s)) for s in key_sizes]
        ax3.scatter([sizes[i] for i in key_indices],
                    [density_list[i] for i in key_indices],
                    color=color, s=100, marker=marker, zorder=5, edgecolors='black', linewidths=1.5)

    ax3.set_xlabel('体素尺寸 (voxels)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('颗粒密度 (颗粒/10000体素³)', fontsize=14, fontweight='bold')
    ax3.set_title('3D颗粒密度 vs 体素尺寸', fontsize=16, fontweight='bold', pad=15)
    ax3.legend(fontsize=12, loc='upper right', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim([0, 520])
    plt.tight_layout()
    plt.savefig('3d_figure3_particle_density.png', dpi=300, bbox_inches='tight')
    print("3D图3已保存: 3d_figure3_particle_density.png")
    plt.close()

    # 图4: 数据表格(扩展到1024)
    fig4, ax4 = plt.subplots(figsize=(11, 9))
    ax4.axis('off')

    table_data = []
    table_data.append(['体素尺寸', '松散\n(φ=50%)', '随机\n(φ=40%)', '中等\n(φ=35%)', '密实FCC\n(φ=26%)'])

    key_sizes_table = [16, 32, 64, 128, 256, 512, 1024]
    for size in key_sizes_table:
        row = [f'{size}³']
        for ratio in packing_ratios:
            gen = Voronoi3DParticleGenerator(size)
            max_p = gen.calculate_max_particles(size, min_diameter=3, packing_ratio=ratio)
            if max_p >= 1000000:
                row.append(f'{max_p / 1e6:.2f}M')
            elif max_p >= 1000:
                row.append(f'{max_p / 1e3:.1f}K')
            else:
                row.append(f'{max_p}')
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

    ax4.set_title('3D最大颗粒数量对照表\n(基于最小颗粒直径3体素)',
                  fontsize=16, fontweight='bold', pad=30)

    # 添加说明文字
    fig4.text(0.5, 0.05,
              'φ = Porosity\n' +
              'K = 千, M = 百万 | FCC = 面心立方最密堆积(74%)',
              ha='center', fontsize=11, style='italic',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, pad=0.8))

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig('3d_figure4_data_table.png', dpi=300, bbox_inches='tight')
    print("3D图4已保存: 3d_figure4_data_table.png")
    plt.close()


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 90)
    print("3D GAN-VORONOI POROUS MEDIA GENERATION SYSTEM")
    print("=" * 90)

    # 显示3D最大颗粒容量表
    calculate_max_particles_3d_table()

    # 绘制3D关系曲线
    print("\n\n正在生成3D关系曲线...")
    fig_curve, sizes, all_data = plot_max_particles_3d_vs_size()
    plt.savefig('3d_max_particles_vs_size_combined.png', dpi=200, bbox_inches='tight')
    print("3D组合图已保存: 3d_max_particles_vs_size_combined.png")
    plt.show()

    # 分别保存四个独立图表
    print("\n\n正在生成独立的3D容量图表...")
    plot_individual_3d_figures(sizes, all_data)
    print("\n所有3D容量独立图表已保存完成!")

    # ========== 关键参数配置区域 ==========
    # 在这里修改参数
    vol_size = 128  # 体素空间尺寸 (128³ 或 256³)
    n_particles = 12000  # 颗粒数量
    porosity = 0.30  # 目标孔隙率 (0-1)
    mean_diameter = 8  # 平均直径(体素), None=自动计算, 或指定如 6.0
    std_ratio = 0.3  # 直径变异系数 (0.2-0.4)
    distribution = 'lognormal'  # 分布类型: 'lognormal', 'gamma', 'weibull', 'normal'

    # 参数验证和建议
    print("\n" + "=" * 90)
    print("参数配置检查")
    print("=" * 90)
    print(f"体素空间: {vol_size}³ = {vol_size ** 3:,} voxels")
    print(f"颗粒数量: {n_particles}")
    print(f"目标孔隙率: {porosity:.1%}")

    # 计算理论平均直径
    if mean_diameter is None:
        effective_volume = vol_size ** 3 * (1 - porosity)
        theoretical_diameter = ((6 * effective_volume / n_particles / np.pi) ** (1 / 3)) * 2
        print(f"自动计算平均直径: {theoretical_diameter:.2f} voxels")
    else:
        theoretical_diameter = mean_diameter
        print(f"指定平均直径: {mean_diameter:.2f} voxels")

    # 计算理论容纳能力
    single_particle_volume = (4 / 3) * np.pi * (theoretical_diameter / 2) ** 3
    theoretical_max = int((vol_size ** 3 * (1 - porosity)) / single_particle_volume)
    usage_ratio = n_particles / theoretical_max if theoretical_max > 0 else 0

    print(f"\n理论分析:")
    print(f"  单颗粒平均体积: {single_particle_volume:.2f} voxels³")
    print(f"  理论最大颗粒数: {theoretical_max}")
    print(f"  当前颗粒数占比: {usage_ratio:.1%}")

    if usage_ratio > 1.2:
        print(f"  ⚠️  警告: 颗粒数量过多! 建议减少到 {theoretical_max} 颗或更少")
    elif usage_ratio > 0.8:
        print(f"  ✓ 参数合理")
    else:
        print(f"  ℹ️  提示: 颗粒较少,实际孔隙率可能高于目标值")

    print("=" * 90)
    # ====================================

    # 生成3D示例
    print(f"\n正在生成 {vol_size}×{vol_size}×{vol_size} 3D多孔介质样本...")

    generator = PorousMedia3DGenerator(vol_size)
    results = generator.generate(
        n_particles=n_particles,
        porosity=porosity,
        mean_diameter=mean_diameter,
        std_ratio=std_ratio,
        distribution=distribution
    )

    # 打印分析结果
    print("\n" + "=" * 90)
    print("3D GEOSTATISTICAL ANALYSIS RESULTS")
    print("=" * 90)

    print(f"\nVolume Size: {vol_size}³ voxels")
    print(f"Number of Particles: {len(results['points'])}")
    print(f"Porosity: {results['porosity']:.4f}")
    print(f"Number of 3D pores: {results['n_gaps']}")
    print(f"Maximum pore width: {results['max_width']:.2f} voxels")
    print(f"Percolation (z-direction): {'Yes' if results['percolates'] else 'No'}")
    if results['percolates']:
        print(f"Percolation ratio: {results['percolation_ratio']:.4f}")
    print(f"Hydraulic conductivity estimate: {results['hydraulic_conductivity']:.2e}")

    print("\n3D Particle Size Statistics:")
    stats = results['particle_stats']
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

    # 分别保存六张独立图片
    print("\n正在生成并保存6张独立的3D分析视图...")
    save_individual_3d_views(results, vol_size)

    print("\n" + "=" * 90)
    print("程序执行完成! 共生成10张图片:")
    print("  容量分析: 5张 (组合图 + 4张独立图)")
    print("  样本分析: 6张 (独立视图)")
    print("=" * 90)