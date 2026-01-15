import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage import morphology, measure
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import stats
import warnings
import os
from tqdm import tqdm

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
    """生成器 - 接收Voronoi特征图并生成精细化的多孔介质"""

    def __init__(self, input_channels=2, img_size=256):
        """
        Args:
            input_channels: 输入通道数 (Voronoi基础图 + 噪声通道)
            img_size: 图像尺寸
        """
        super().__init__()

        # 编码器：提取Voronoi特征
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            AttentionBlock(256),

            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )

        # 解码器：生成精细化的多孔介质
        self.decoder = nn.Sequential(
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

            nn.Conv2d(64, 1, 3, padding=1),
            nn.Tanh()  # 输出范围[-1, 1]
        )

    def forward(self, voronoi_img, noise=None):
        """
        Args:
            voronoi_img: Voronoi基础图 [B, 1, H, W]
            noise: 随机噪声 [B, 1, H, W]，如果为None则自动生成
        """
        if noise is None:
            noise = torch.randn_like(voronoi_img)

        # 拼接Voronoi图和噪声
        x = torch.cat([voronoi_img, noise], dim=1)

        # 编码-解码
        features = self.encoder(x)
        output = self.decoder(features)

        return output


class Discriminator(nn.Module):
    """判别器 - 区分真实和GAN生成的多孔介质"""

    def __init__(self, img_size=256):
        super().__init__()

        self.model = nn.Sequential(
            # 输入: [B, 1, 256, 256]
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


# ==================== Voronoi颗粒生成器（作为GAN的输入） ====================
class VoronoiParticleGenerator:
    """Voronoi颗粒生成器 - 生成基础结构供GAN精细化"""

    def __init__(self, img_size=256):
        self.img_size = img_size

    def calculate_parameters(self, img_size, target_porosity=0.3,
                             n_particles=None, avg_diameter=None, min_diameter=3):
        """智能计算生成参数（三选二模式）"""
        total_area = img_size ** 2

        params_provided = sum([
            target_porosity is not None,
            n_particles is not None,
            avg_diameter is not None
        ])

        if params_provided < 2:
            raise ValueError(
                "必须提供以下三个参数中的任意两个：target_porosity, n_particles, avg_diameter"
            )

        if target_porosity is not None and n_particles is not None and avg_diameter is None:
            particle_area = total_area * (1 - target_porosity)
            avg_particle_area = particle_area / n_particles
            avg_diameter = 2 * np.sqrt(avg_particle_area / np.pi)

        elif target_porosity is not None and avg_diameter is not None and n_particles is None:
            particle_area = total_area * (1 - target_porosity)
            avg_particle_area = np.pi * (avg_diameter / 2) ** 2
            n_particles = int(particle_area / avg_particle_area)

        elif n_particles is not None and avg_diameter is not None and target_porosity is None:
            avg_particle_area = np.pi * (avg_diameter / 2) ** 2
            total_particle_area = n_particles * avg_particle_area
            target_porosity = 1 - (total_particle_area / total_area)

            if target_porosity < 0:
                target_porosity = 0.05

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

        diameters = np.clip(diameters, mean_diameter * 0.3, mean_diameter * 3)
        return diameters

    def create_voronoi_base(self, n_particles=None, target_porosity=0.3,
                            avg_diameter=None, distribution='lognormal',
                            erosion_factor=0.05, min_diameter=3):
        """
        创建Voronoi基础图（供GAN精细化）

        返回:
            base_image: 基础Voronoi图 [0, 1]
            target_image: 目标真实图（用于训练）[0, 1]
            actual_porosity: 实际孔隙率
        """
        n_particles, avg_diameter, target_porosity = self.calculate_parameters(
            self.img_size, target_porosity, n_particles, avg_diameter, min_diameter
        )

        compensation_factor = 1.0 / (1 - erosion_factor * 2)
        compensated_diameter = avg_diameter * np.sqrt(compensation_factor)

        diameters = self.generate_particle_diameters(n_particles, compensated_diameter,
                                                     distribution=distribution)

        points = np.random.rand(n_particles, 2) * self.img_size

        # 创建基础Voronoi图（粗糙版本，作为GAN输入）
        base_image = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        # 创建目标图像（精细版本，作为训练目标）
        target_image = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        for point, diameter in zip(points, diameters):
            radius = diameter / 2
            y, x = np.ogrid[:self.img_size, :self.img_size]
            center_x, center_y = point[0], point[1]
            dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            # 基础版本：简单的圆形，无侵蚀
            base_mask = dist_from_center < radius
            base_image[base_mask] = 1

            # 目标版本：带侵蚀和细节
            target_mask = dist_from_center < radius * (1 - erosion_factor)
            target_image[target_mask] = 1

        # 基础图：轻微模糊
        base_image = gaussian_filter(base_image.astype(float), sigma=0.3)

        # 目标图：细节保留
        target_image = gaussian_filter(target_image.astype(float), sigma=0.5)

        actual_porosity = (target_image < 0.5).sum() / target_image.size

        return base_image, target_image, actual_porosity


# ==================== 数据集类 ====================
class VoronoiGANDataset(Dataset):
    """Voronoi-GAN配对数据集"""

    def __init__(self, base_images, target_images):
        """
        Args:
            base_images: Voronoi基础图列表
            target_images: 目标精细图列表
        """
        self.base_images = base_images
        self.target_images = target_images

    def __len__(self):
        return len(self.base_images)

    def __getitem__(self, idx):
        base = self.base_images[idx]
        target = self.target_images[idx]

        # 归一化到[-1, 1]
        base = (base - 0.5) / 0.5
        target = (target - 0.5) / 0.5

        # 转换为tensor
        base_tensor = torch.FloatTensor(base).unsqueeze(0)
        target_tensor = torch.FloatTensor(target).unsqueeze(0)

        return base_tensor, target_tensor


# ==================== GAN训练器（Voronoi-GAN） ====================
class VoronoiGANTrainer:
    """Voronoi-GAN训练器：将粗糙Voronoi图精细化"""

    def __init__(self, img_size=256, device=None):
        self.img_size = img_size
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化网络
        self.generator = Generator(input_channels=2, img_size=img_size).to(self.device)
        self.discriminator = Discriminator(img_size).to(self.device)

        # 损失函数
        self.adversarial_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()  # 用于保持结构相似性

        # 优化器
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # 训练历史
        self.g_losses = []
        self.d_losses = []
        self.l1_losses = []

    def train(self, dataloader, epochs=100, lambda_l1=100, save_interval=10, save_dir='voronoi_gan_checkpoints'):
        """
        训练Voronoi-GAN

        Args:
            dataloader: 数据加载器 (base_image, target_image)
            epochs: 训练轮数
            lambda_l1: L1损失权重
            save_interval: 保存间隔
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        print(f"开始训练Voronoi-GAN on {self.device}")
        print(f"训练轮数: {epochs}, 批次大小: {dataloader.batch_size}")
        print(f"L1损失权重: {lambda_l1}")
        print("=" * 80)

        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            epoch_l1_loss = 0

            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}')

            for i, (base_imgs, target_imgs) in enumerate(progress_bar):
                batch_size = base_imgs.size(0)
                base_imgs = base_imgs.to(self.device)
                target_imgs = target_imgs.to(self.device)

                # 真实和虚假标签
                valid = torch.ones(batch_size, 1).to(self.device)
                fake = torch.zeros(batch_size, 1).to(self.device)

                # ---------------------
                #  训练生成器
                # ---------------------
                self.optimizer_G.zero_grad()

                # 生成精细化图像
                gen_imgs = self.generator(base_imgs)

                # 生成器损失 = 对抗损失 + L1损失（结构保持）
                g_adv_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
                g_l1_loss = self.l1_loss(gen_imgs, target_imgs)
                g_loss = g_adv_loss + lambda_l1 * g_l1_loss

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  训练判别器
                # ---------------------
                self.optimizer_D.zero_grad()

                # 真实图像损失
                real_loss = self.adversarial_loss(self.discriminator(target_imgs), valid)
                # 虚假图像损失
                fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
                # 总判别器损失
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()

                # 记录损失
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                epoch_l1_loss += g_l1_loss.item()

                # 更新进度条
                progress_bar.set_postfix({
                    'D_loss': f'{d_loss.item():.4f}',
                    'G_loss': f'{g_loss.item():.4f}',
                    'L1_loss': f'{g_l1_loss.item():.4f}'
                })

            # 计算平均损失
            avg_g_loss = epoch_g_loss / len(dataloader)
            avg_d_loss = epoch_d_loss / len(dataloader)
            avg_l1_loss = epoch_l1_loss / len(dataloader)

            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)
            self.l1_losses.append(avg_l1_loss)

            print(f"Epoch [{epoch + 1}/{epochs}] - D: {avg_d_loss:.4f}, G: {avg_g_loss:.4f}, L1: {avg_l1_loss:.4f}")

            # 定期保存检查点和生成样本
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1, save_dir)
                self.generate_comparison(base_imgs[:4], target_imgs[:4], epoch + 1, save_dir)

        print("\n训练完成!")
        self.plot_training_history(save_dir)

    def save_checkpoint(self, epoch, save_dir):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'l1_losses': self.l1_losses,
        }
        path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)
        print(f"✓ 检查点已保存: {path}")

    def load_checkpoint(self, path):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.g_losses = checkpoint['g_losses']
        self.d_losses = checkpoint['d_losses']
        self.l1_losses = checkpoint['l1_losses']
        print(f"✓ 检查点已加载: {path}")
        return checkpoint['epoch']

    def generate_comparison(self, base_imgs, target_imgs, epoch, save_dir):
        """生成对比图：Voronoi基础 vs GAN生成 vs 目标"""
        self.generator.eval()
        with torch.no_grad():
            gen_imgs = self.generator(base_imgs)

        # 转换为numpy并反归一化
        base_np = (base_imgs.cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        gen_np = (gen_imgs.cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        target_np = (target_imgs.cpu().numpy() * 0.5 + 0.5).clip(0, 1)

        # 可视化
        n_samples = min(4, base_imgs.size(0))
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 3))

        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            axes[i, 0].imshow(base_np[i, 0], cmap='gray')
            axes[i, 0].set_title('Voronoi Base')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(gen_np[i, 0], cmap='gray')
            axes[i, 1].set_title('GAN Generated')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(target_np[i, 0], cmap='gray')
            axes[i, 2].set_title('Target')
            axes[i, 2].axis('off')

        plt.suptitle(f'Voronoi-GAN Results - Epoch {epoch}', fontsize=16)
        plt.tight_layout()
        path = os.path.join(save_dir, f'comparison_epoch_{epoch}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

        self.generator.train()

    def plot_training_history(self, save_dir):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # 对抗损失
        axes[0].plot(self.g_losses, label='Generator Loss', alpha=0.7)
        axes[0].plot(self.d_losses, label='Discriminator Loss', alpha=0.7)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Adversarial Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # L1损失
        axes[1].plot(self.l1_losses, label='L1 Loss', color='green', alpha=0.7)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('L1 Loss (Structure Preservation)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(save_dir, 'training_history.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 训练历史已保存: {path}")


# ==================== 主系统：Voronoi-GAN集成 ====================
class PorousMediaGenerator:
    """多孔材料生成主系统 - Voronoi与GAN结合"""

    def __init__(self, img_size=256, gan_checkpoint=None):
        self.img_size = img_size
        self.voronoi_gen = VoronoiParticleGenerator(img_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化GAN
        self.gan_trainer = VoronoiGANTrainer(img_size=img_size, device=self.device)

        # 如果提供了检查点，加载训练好的模型
        if gan_checkpoint:
            self.gan_trainer.load_checkpoint(gan_checkpoint)
            self.is_trained = True
        else:
            self.is_trained = False

    def generate_training_data(self, n_samples=1000, particle_range=(500, 4000),
                               diameter_range=(4, 10)):
        """
        生成训练数据：Voronoi基础图 + 目标图
        使用模式3：指定颗粒数 + 平均直径 → 自动计算孔隙率

        Args:
            n_samples: 生成样本数量
            particle_range: 颗粒数范围 (min, max)
            diameter_range: 平均直径范围 (min, max) 单位：像素

        Returns:
            base_images: Voronoi基础图列表
            target_images: 目标精细图列表
        """
        print(f"正在生成 {n_samples} 个训练样本（模式3: 颗粒数+直径 → 孔隙率）...")
        print(f"  颗粒数范围: {particle_range[0]}-{particle_range[1]}")
        print(f"  直径范围: {diameter_range[0]}-{diameter_range[1]} 像素")

        base_images = []
        target_images = []
        porosity_stats = []

        for i in tqdm(range(n_samples)):
            # 随机选择颗粒数和直径
            n_particles = np.random.randint(particle_range[0], particle_range[1])
            avg_diameter = np.random.uniform(*diameter_range)

            base_img, target_img, actual_porosity = self.voronoi_gen.create_voronoi_base(
                n_particles=n_particles,
                avg_diameter=avg_diameter,
                erosion_factor=np.random.uniform(0.02, 0.08)
            )

            base_images.append(base_img)
            target_images.append(target_img)
            porosity_stats.append(actual_porosity)

        # 打印统计信息
        porosity_stats = np.array(porosity_stats)
        print(f"\n✓ 训练数据生成完成")
        print(f"  孔隙率统计: 最小={porosity_stats.min():.2%}, "
              f"最大={porosity_stats.max():.2%}, "
              f"平均={porosity_stats.mean():.2%}")

        return base_images, target_images

    def train_gan(self, n_training_samples=1000, batch_size=8, epochs=100,
                  lambda_l1=100, particle_range=(500, 4000), diameter_range=(4, 10),
                  save_dir='voronoi_gan_checkpoints'):
        """
        训练Voronoi-GAN模型
        使用模式3：指定颗粒数 + 平均直径 → 自动计算孔隙率

        Args:
            n_training_samples: 训练样本数量
            batch_size: 批次大小
            epochs: 训练轮数
            lambda_l1: L1损失权重（控制结构保持）
            particle_range: 颗粒数范围 (min, max)
            diameter_range: 平均直径范围 (min, max)
            save_dir: 保存目录
        """
        # 生成训练数据
        base_images, target_images = self.generate_training_data(
            n_training_samples,
            particle_range=particle_range,
            diameter_range=diameter_range
        )

        # 创建数据集和数据加载器
        dataset = VoronoiGANDataset(base_images, target_images)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        # 训练GAN
        self.gan_trainer.train(dataloader, epochs=epochs, lambda_l1=lambda_l1, save_dir=save_dir)

        self.is_trained = True
        print("✓ Voronoi-GAN训练完成")

    def generate(self, n_particles=None, porosity=None, avg_diameter=None,
                 use_gan=True, distribution='lognormal', erosion_factor=0.05):
        """
        生成多孔介质图像

        Args:
            n_particles: 颗粒数量（三选二）
            porosity: 孔隙率（三选二）
            avg_diameter: 平均直径（三选二）
            use_gan: 是否使用GAN精细化
            distribution: 粒径分布类型
            erosion_factor: 侵蚀因子

        Returns:
            如果use_gan=False: 返回(Voronoi目标图, 实际孔隙率)
            如果use_gan=True: 返回(Voronoi基础图, GAN精细化图, 实际孔隙率)
        """
        # 生成Voronoi基础图和目标图
        base_img, target_img, actual_porosity = self.voronoi_gen.create_voronoi_base(
            n_particles=n_particles,
            target_porosity=porosity,
            avg_diameter=avg_diameter,
            distribution=distribution,
            erosion_factor=erosion_factor
        )

        if not use_gan:
            # 不使用GAN，返回原始Voronoi目标图
            return target_img, actual_porosity

        # 使用GAN精细化
        if not self.is_trained:
            print("警告: GAN未训练，返回原始Voronoi图")
            return base_img, target_img, actual_porosity

        self.gan_trainer.generator.eval()
        with torch.no_grad():
            # 准备输入
            base_tensor = torch.FloatTensor((base_img - 0.5) / 0.5).unsqueeze(0).unsqueeze(0).to(self.device)

            # GAN生成
            gan_output = self.gan_trainer.generator(base_tensor)
            gan_img = gan_output.cpu().numpy()[0, 0]

            # 反归一化
            gan_img = (gan_img * 0.5 + 0.5).clip(0, 1)

        # 返回三元组：基础图、GAN生成图、实际孔隙率
        return base_img, gan_img, actual_porosity


# ==================== 可视化函数 ====================
def visualize_comparison(base_img, gan_img, porosity):
    """可视化Voronoi基础图与GAN精细化后的对比"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(base_img, cmap='gray')
    axes[0].set_title('Voronoi Base Image\n(Coarse Structure)', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(gan_img, cmap='gray')
    axes[1].set_title(f'GAN Refined Image\n(Porosity: {porosity:.2%})', fontsize=12)
    axes[1].axis('off')

    # 差异图
    diff = np.abs(gan_img - base_img)
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Difference Map\n(GAN Refinement)', fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    return fig


def demo_training():
    """演示Voronoi-GAN训练流程"""
    print("\n" + "=" * 80)
    print("Voronoi-GAN 训练演示（模式3: 颗粒数+直径 → 孔隙率）")
    print("=" * 80)

    # 创建生成器
    generator = PorousMediaGenerator(img_size=256)

    # 训练GAN（小规模演示）
    print("\n步骤1: 训练Voronoi-GAN模型...")
    print("说明: GAN会学习如何将粗糙的Voronoi基础图精细化")
    print("训练策略: 指定颗粒数和直径，让系统自动计算最终孔隙率")

    generator.train_gan(
        n_training_samples=1200,  # 实际应用建议1000+
        batch_size=8,
        epochs=150,  # 实际应用建议100+
        lambda_l1=100,  # L1损失权重，控制结构保持
        particle_range=(800, 3000),  # 颗粒数范围
        diameter_range=(4, 10),  # 直径范围
        save_dir='voronoi_gan_demo'
    )

    # 使用GAN生成
    print("\n步骤2: 使用训练好的GAN生成精细化图像...")

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # 生成不同参数组合的样本
    particle_counts = [1000, 2000, 3000]
    diameters = [5, 7, 9]

    for i, n_particles in enumerate(particle_counts):
        for j, diameter in enumerate(diameters):
            base_img, gan_img, actual_por = generator.generate(
                n_particles=n_particles,
                avg_diameter=diameter,
                use_gan=True
            )

            # 显示GAN精细化后的图像
            axes[i, j].imshow(gan_img, cmap='gray')
            axes[i, j].set_title(
                f'N={n_particles}, D={diameter}px\nPor: {actual_por:.1%}',
                fontsize=10
            )
            axes[i, j].axis('off')

    plt.suptitle('Voronoi-GAN Generated (Particles + Diameter → Porosity)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('voronoi_gan_results_mode3.png', dpi=150, bbox_inches='tight')
    print("✓ 结果已保存: voronoi_gan_results_mode3.png")
    plt.show()


def demo_comparison():
    """对比Voronoi原始图与GAN精细化后的效果"""
    print("\n" + "=" * 80)
    print("Voronoi vs Voronoi-GAN 对比（模式3: 颗粒数+直径）")
    print("=" * 80)

    generator = PorousMediaGenerator(img_size=256)

    print("\n步骤1: 生成Voronoi基础图（粗糙）...")
    print("参数: 颗粒数=2000, 平均直径=6px")
    voronoi_only, por1 = generator.generate(
        n_particles=2000,
        avg_diameter=6,
        use_gan=False
    )
    print(f"实际孔隙率: {por1:.2%}")

    # 如果GAN已训练，生成精细化版本
    print("\n步骤2: 使用GAN精细化...")
    print("(如果GAN未训练，此步骤将被跳过)")

    try:
        base_img, gan_img, por2 = generator.generate(
            n_particles=2000,
            avg_diameter=6,
            use_gan=True
        )

        fig = visualize_comparison(base_img, gan_img, por2)
        plt.savefig('voronoi_gan_comparison_mode3.png', dpi=150, bbox_inches='tight')
        print("✓ 对比图已保存: voronoi_gan_comparison_mode3.png")
        plt.show()

    except Exception as e:
        print(f"GAN精细化失败（可能未训练）: {e}")
        print("显示纯Voronoi结果...")

        plt.figure(figsize=(8, 8))
        plt.imshow(voronoi_only, cmap='gray')
        plt.title(f'Voronoi Only\nN=2000, D=6px, Porosity: {por1:.2%}')
        plt.axis('off')
        plt.savefig('voronoi_only_mode3.png', dpi=150, bbox_inches='tight')
        print("✓ 结果已保存: voronoi_only_mode3.png")
        plt.show()


def demo_quick_test():
    """快速测试：使用模式3生成不同颗粒数和直径的样本"""
    print("\n" + "=" * 80)
    print("快速测试：模式3（颗粒数+直径 → 孔隙率）")
    print("=" * 80)

    generator = PorousMediaGenerator(img_size=256)

    # 测试不同的颗粒数
    particle_counts = [800, 1200, 1600, 2000, 2500, 3000]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    print("\n固定直径=6px，测试不同颗粒数...")

    for idx, n_particles in enumerate(particle_counts):
        row = idx // 3
        col = idx % 3

        # 生成Voronoi图（不使用GAN）
        img, actual_por = generator.generate(
            n_particles=n_particles,
            avg_diameter=6,
            use_gan=False
        )

        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(
            f'N={n_particles}, D=6px\nPorosity: {actual_por:.1%}',
            fontsize=10
        )
        axes[row, col].axis('off')

        print(f"  N={n_particles} → Porosity={actual_por:.2%}")

    plt.suptitle('Voronoi Generated (Mode 3: Particles+Diameter→Porosity)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('voronoi_mode3_particle_test.png', dpi=150, bbox_inches='tight')
    print("\n✓ 结果已保存: voronoi_mode3_particle_test.png")
    plt.show()

    # 测试不同的直径
    print("\n固定颗粒数=2000，测试不同直径...")
    diameters = [4, 5, 6, 7, 8, 10]

    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))

    for idx, diameter in enumerate(diameters):
        row = idx // 3
        col = idx % 3

        img, actual_por = generator.generate(
            n_particles=2000,
            avg_diameter=diameter,
            use_gan=False
        )

        axes2[row, col].imshow(img, cmap='gray')
        axes2[row, col].set_title(
            f'N=2000, D={diameter}px\nPorosity: {actual_por:.1%}',
            fontsize=10
        )
        axes2[row, col].axis('off')

        print(f"  D={diameter}px → Porosity={actual_por:.2%}")

    plt.suptitle('Voronoi Generated (Mode 3: Varying Diameter)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('voronoi_mode3_diameter_test.png', dpi=150, bbox_inches='tight')
    print("\n✓ 结果已保存: voronoi_mode3_diameter_test.png")
    plt.show()


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Voronoi-GAN 集成多孔介质生成系统")
    print("=" * 80)
    print("\n系统架构:")
    print("  1. Voronoi生成器 → 创建粗糙的基础结构")
    print("  2. GAN生成器 → 接收Voronoi图并精细化")
    print("  3. 判别器 → 确保生成的图像真实")
    print("\n生成模式（模式3）:")
    print("  指定: 颗粒数 + 平均直径")
    print("  自动计算: 孔隙率")
    print("\n工作流程:")
    print("  输入(颗粒数,直径) → Voronoi基础图 → GAN精细化 → 高质量多孔介质")
    print("=" * 80)

    print("\n选择运行模式:")
    print("  [1] 快速测试（生成Voronoi样本，测试模式3）")
    print("  [2] GAN训练演示（完整流程，需要时间）")
    print("  [3] 对比测试（Voronoi vs GAN）")

    # 默认运行快速测试
    print("\n>>> 运行模式1: 快速测试（模式3）")
    demo_quick_test()

    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)
    print("\n参数说明（模式3）:")
    print("  - 颗粒数越多 → 孔隙率越小（颗粒占据更多空间）")
    print("  - 直径越大 → 孔隙率越小（颗粒更大）")
    print("  - 系统自动计算最终孔隙率")
    print("\n如需训练GAN并查看精细化效果，请取消注释:")
    print("# demo_training()")
    print("\n如需对比Voronoi和GAN效果，请取消注释:")
    print("# demo_comparison()")

    # 取消下面的注释来运行完整GAN训练
    demo_training()

    # 取消下面的注释来运行对比测试
    demo_comparison()
