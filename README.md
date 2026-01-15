Voronoi-GAN Porous Media Generator

A sophisticated porous media generation system combining Voronoi tessellation with Generative Adversarial Networks (GAN). The system creates coarse porous structures using Voronoi algorithms, then refines them through GAN to produce high-quality, realistic porous media images.

## 🌟 Features

- **Two-Stage Generation Architecture**: Voronoi base generation + GAN refinement
- **Flexible Parameter Control**: Automatic porosity calculation from particle count and diameter
- **Attention Mechanism**: Integrated Attention Blocks for enhanced generation quality
- **Multiple Grain Size Distributions**: Support for lognormal, Gamma, Weibull, and normal distributions
- **Complete Training Pipeline**: Full workflow including data generation, model training, and visualization

## 📋 Requirements

```bash
Python = 3.10
```

### Dependencies

```bash
numpy
matplotlib
scipy
scikit-image
torch >= 1.7.0
tqdm
```


**Tip**: For first-time users, run `quick_test.py` to understand system functionality before GAN training.
