# Pix2Pix: Generación de Rostros de Gatos desde Landmarks

Implementación de **Pix2Pix** (Conditional GAN) para generar imágenes realistas de rostros de gatos a partir de mapas de calor de landmarks faciales. El proyecto utiliza el dataset **CatFLW** y arquitecturas U-Net + PatchGAN para lograr una traducción imagen-a-imagen de alta calidad.

## 🎯 Objetivo

Transformar representaciones abstractas de landmarks faciales (9 puntos clave: ojos, nariz, boca, orejas) en imágenes fotorrealistas de gatos mediante aprendizaje adversarial condicional.

## 🏗️ Arquitectura

### Generador: U-Net
- **Input**: 1 canal (heatmap) → **Output**: 3 canales (RGB)
- **Encoder**: 8 capas de downsampling (64 → 512 canales)
- **Bottleneck**: Capa de 512 canales en resolución 1×1
- **Decoder**: 7 capas de upsampling con skip connections U-Net
- **Normalización**: InstanceNorm2d (mejor estabilidad que BatchNorm)
- **Dropout**: 50% en las primeras 3 capas del decoder
- **Salida**: Tanh activation para normalización [-1, 1]

### Discriminador: PatchGAN
- **Input**: 4 canales (heatmap 1ch + imagen RGB 3ch concatenados)
- Clasificador de parches 30×30 para evaluar detalles locales
- 5 capas convolucionales (64 → 512 canales)
- **Normalización**: InstanceNorm2d (excepto primera capa)
- Salida sin activación (usa MSELoss directamente)

## 📊 Dataset: CatFLW

- **Total**: 2,079 imágenes de rostros de gatos
- **Split**: 1,663 train / 207 val / 209 test (~80%/10%/10%)
- **Resolución**: 256×256 píxeles
- **Landmarks**: 9 puntos faciales clave (ojos, nariz, boca, orejas)
- **Formato**:
  - **A (Input)**: Mapas de calor Gaussianos de 1 canal (.npy, uint8)
  - **B (Target)**: Imágenes RGB (.jpg, calidad 95)
- **Preprocesamiento**: 
  - Adaptive bounding box con expansión dinámica
  - Mapas de calor Gaussianos (σ configurable, default=2.0)
  - Normalización [-1, 1] para ambos A y B
- **Augmentation** (solo training):
  - Flip horizontal sincronizado entre A y B
  - ColorJitter ligero (brightness, contrast, saturation, hue)

## 🔧 Configuración de Entrenamiento

### Parámetros Actuales
```python
epochs = 100
batch_size = 24
lambda_L1 = 150         # Pérdida de reconstrucción L1
lambda_perc = 2         # Pérdida perceptual VGG16
lr = 0.0002             # Learning rate Generator
lr_D = 0.0001           # Learning rate Discriminator (50% de lr_G)
betas = (0.5, 0.999)
```

### Funciones de Pérdida
1. **Adversarial Loss**: MSELoss (LSGAN) sin label smoothing
   - Real labels: 1.0
   - Fake labels: 0.0
   - Más estable que BCEWithLogitsLoss para este caso
2. **L1 Loss**: Reconstrucción pixel-a-pixel (λ=150)
3. **Perceptual Loss**: Features VGG16 (layers [:16], λ=2)

### Técnicas de Optimización
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=50, T_mult=2)
- **Discriminador**: Entrenado en cada batch (no cada N batches)
- **Selección de modelo**: Solo por Val L1 (PSNR solo informativo)
- **Checkpointing**: 
  - `last_checkpoint.pth`: Guarda estado completo cada época
  - `G_best.pth` / `D_best.pth`: Guarda cuando mejora Val L1
- **Resume automático**: Si existe checkpoint, continúa desde ahí
- **Data Augmentation** (solo training):
  - HorizontalFlip sincronizado (50%)
  - ColorJitter muy ligero (brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01)

## 📈 Resultados (Época ~70)

| Métrica | Valor | Objetivo |
|---------|-------|----------|
| **Loss D** | ~0.47 | 0.3-0.7 ✅ |
| **Loss G** | ~30 | 20-40 ✅ |
| **Val L1** | ~0.49 | <0.20 ⚠️ |
| **PSNR** | ~4-5 dB | >20 dB ⚠️ |

### Estado Actual
- ✅ **Equilibrio adversarial estable** (D y G convergen sin colapso)
- ✅ **Estructura facial correcta** (landmarks → posición facial precisa)
- ✅ **InstanceNorm**: Mejor estabilidad que BatchNorm
- ✅ **Checkpoint resume**: Funcional y probado
- ⚠️ **Blur moderado**: Detalles finos suavizados (bigotes, textura de pelaje)

### Observaciones
- `lambda_L1=150` alto → prioriza reconstrucción promedio sobre detalles
- PSNR usado solo como métrica descriptiva, no influye en entrenamiento
- Val L1 es la única métrica para selección del mejor modelo

## 🚀 Uso

### Instalación
```bash
pip install -r requirements.txt
```

O manualmente:
```bash
pip install torch torchvision opencv-python numpy pillow tqdm matplotlib
```

### Preparar Dataset
Usando el script `generate_dataset.py`:
```bash
python generate_dataset.py \
    --input_root "CatFLW dataset" \
    --output_root "datasets/catflw" \
    --sigma 2.0 \
    --seed 42
```

O desde Python:
```python
from generate_dataset import prepare_catflw_dataset

prepare_catflw_dataset(
    input_root="CatFLW dataset",
    output_root="datasets/catflw",
    sigma=2.0,
    min_coverage=0.9
)
```

### Entrenar Modelo
```python
# Entrenamiento desde cero
G = train_pix2pix(
    dataset_root="datasets/catflw",
    epochs=100,
    batch_size=24,
    lambda_L1=150,
    lambda_perc=2,
    lr=0.0002,
    resume=False  # False para empezar desde cero
)

# Reanudar desde checkpoint (automático)
G = train_pix2pix(
    dataset_root="datasets/catflw",
    epochs=100,
    batch_size=24,
    lambda_L1=150,
    lambda_perc=2,
    lr=0.0002,
    resume=True  # True por defecto
)
```

### Evaluar en Test
```python
# Evaluar con mejor modelo guardado
avg_l1, avg_psnr = evaluate_on_test(G, dataset_root="datasets/catflw/test")

# Mostrar mejores ejemplos visuales
test_loader = DataLoader(
    Pix2PixDataset("datasets/catflw/test"), 
    batch_size=24,
    shuffle=False
)
show_samples(G, test_loader, device, n=16, 
            title="Mejores Resultados", 
            save_path="results/best_samples_test.png")
```

### Reanudar Entrenamiento
**Automático**: Si `resume=True` (default) y existe `checkpoints/last_checkpoint.pth`, el entrenamiento continúa desde la última época guardada, preservando:
- Estado de G y D
- Estado de optimizadores (momentum, etc.)
- Mejor Val L1 registrado
- Número de época

## 📁 Estructura del Proyecto

```
pix2pix/
├── pix2pix_proyect.ipynb      # Notebook principal con todo el pipeline
├── generate_dataset.py         # Script de preprocesamiento CatFLW → Pix2Pix
├── README.md                   # Este archivo
├── requirements.txt            # Dependencias Python
├── .gitignore                  # Configuración de exclusiones
├── CatFLW dataset/            # Dataset original (NO versionado)
│   ├── images/                # 2,079 imágenes .png originales
│   └── labels/                # 2,079 archivos .json con landmarks
├── datasets/                   # Dataset procesado (NO versionado)
│   └── catflw/
│       ├── train/             # 1,663 pares
│       │   ├── A/             # Heatmaps .npy (1 canal)
│       │   └── B/             # Imágenes .jpg (RGB)
│       ├── val/               # 207 pares
│       │   ├── A/
│       │   └── B/
│       └── test/              # 209 pares
│           ├── A/
│           └── B/
├── checkpoints/               # Modelos guardados
│   ├── G_best.pth            # Mejor generador por Val L1
│   ├── D_best.pth            # Mejor discriminador
│   └── last_checkpoint.pth   # Último estado completo (NO versionado)
└── results/                   # Visualizaciones generadas
    ├── dataset_samples.png    # Ejemplos del dataset
    ├── epoch_{N}.png          # Progreso cada 10 épocas (NO versionadas)
    └── best_samples_test.png  # Mejores resultados en test
```

## 🔄 Próximos Pasos

### Optimización para Mejorar Detalles
1. **Reducir lambda_L1**: 150 → 100 → 60 (menos over-smoothing)
2. **Reducir lambda_perc**: 2 → 1 (menos dependencia de VGG)
3. **Sigma más definido**: Regenerar dataset con σ=1.5 (landmarks más sharp)
4. **Continuar entrenamiento**: 100 → 200 épocas
5. **Evaluar arquitectura**: Considerar attention mechanisms

### Expectativas con Ajustes
- **Val L1**: 0.15-0.25
- **PSNR**: 15-25 dB
- **Calidad visual**: Mejor definición en bigotes, pelaje y ojos

## 📝 Historial de Versiones

### v2.0 - Noviembre 2025 (Actual)
- ✅ Implementado checkpoint resume automático
- ✅ Simplificado pipeline: sin early stopping, sin PSNR en entrenamiento
- ✅ InstanceNorm en lugar de BatchNorm para mejor estabilidad
- ✅ Selección de mejor modelo solo por Val L1
- ✅ Script `generate_dataset.py` con adaptive bounding box
- ✅ README actualizado con documentación completa
- 🔄 Entrenamiento en progreso (~70 épocas completadas)

### v1.0 - Baseline
- Entrenamiento inicial hasta época 200
- Lambda ajustado en época 101 (100/5 → 60/3)
- Identificado problema de blur por lambda_L1 alto

## 🛠️ Tecnologías

- **PyTorch 2.x**: Framework de deep learning
- **CUDA**: Aceleración GPU (si disponible)
- **torchvision**: Transformaciones y VGG16 preentrenado para perceptual loss
- **OpenCV**: Procesamiento de imágenes en preprocesamiento
- **NumPy**: Operaciones de array y carga de heatmaps .npy
- **Pillow (PIL)**: Carga y manipulación de imágenes
- **tqdm**: Barras de progreso
- **matplotlib**: Visualización de resultados

## 👨‍💻 Autor

**Samuel Ordaz**  
📧 samuel.ordaz@cimat.mx

## 📄 Licencia

Este proyecto es parte de investigación académica en Deep Learning.

## 🙏 Referencias

- Isola, P., et al. (2017). "Image-to-Image Translation with Conditional Adversarial Networks" (Pix2Pix paper)
- Ronneberger, O., et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- Dataset CatFLW: Facial Landmarks in the Wild for Cats
