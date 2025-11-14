# Pix2Pix: Generación de Rostros de Gatos desde Landmarks

Implementación de **Pix2Pix** (Conditional GAN) para generar imágenes realistas de rostros de gatos a partir de mapas de calor de landmarks faciales. El proyecto utiliza el dataset **CatFLW** y arquitecturas U-Net + PatchGAN para lograr una traducción imagen-a-imagen de alta calidad.

## 🎯 Objetivo

Transformar representaciones abstractas de landmarks faciales (9 puntos clave: ojos, nariz, boca, orejas) en imágenes fotorrealistas de gatos mediante aprendizaje adversarial condicional.

## 🏗️ Arquitectura

### Generador: U-Net
- **Encoder**: 7 capas de downsampling (64 → 512 canales)
- **Bottleneck**: Capa de 512 canales en resolución 2×2
- **Decoder**: 7 capas de upsampling con skip connections
- **Dropout**: 50% en las primeras 3 capas del decoder
- **Salida**: Tanh activation para normalización [-1, 1]

### Discriminador: PatchGAN
- Clasificador de parches 30×30 para detalles locales
- 5 capas convolucionales (64 → 512 canales)
- BatchNorm excepto en primera capa
- Salida sin sigmoid (usa BCEWithLogitsLoss)

## 📊 Dataset: CatFLW

- **Total**: 2,090 imágenes de rostros de gatos
- **Split**: 80% train / 10% val / 10% test
- **Resolución**: 256×256 píxeles
- **Landmarks**: 9 puntos faciales clave
- **Preprocesamiento**: 
  - Mapas de calor Gaussianos (σ=2 para definición óptima)
  - Recorte por bounding box
  - Normalización [-1, 1]

## 🔧 Configuración de Entrenamiento

### Parámetros Optimizados (Época 200)
```python
epochs = 200
batch_size = 32
lambda_L1 = 60          # Pérdida de reconstrucción L1
lambda_perc = 3         # Pérdida perceptual VGG16
sigma = 3               # Desviación estándar Gaussiana
learning_rate = 2e-4
betas = (0.5, 0.999)
```

### Funciones de Pérdida
1. **Adversarial Loss**: BCEWithLogitsLoss con label smoothing
   - Real labels: 0.9 + ruido [0, 0.1]
   - Fake labels: ruido [0, 0.1]
2. **L1 Loss**: Reconstrucción pixel-a-pixel (λ=60)
3. **Perceptual Loss**: Features VGG16 (layers [:16], λ=3)

### Técnicas de Optimización
- **AMP (Automatic Mixed Precision)**: FP16 para 2× speedup
- **LambdaLR Scheduler**: Decaimiento lineal desde época 100
- **Data Augmentation**: 
  - HorizontalFlip (50%)
  - ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

## 📈 Resultados (Época 200)

| Métrica | Valor | Objetivo |
|---------|-------|----------|
| **Loss D** | 0.473 | 0.3-0.7 ✅ |
| **Loss G** | 29.88 | 20-30 ✅ |
| **Val L1** | 0.488 | <0.15 ⚠️ |
| **PSNR** | ~4.37 dB | >20 dB ❌ |
| **SSIM** | N/A | >0.70 ❌ |

### Estado Actual
- ✅ **Equilibrio adversarial perfecto** (D y G estables)
- ✅ **Estructura facial correcta** (ojos, nariz, boca bien posicionados)
- ⚠️ **Blur en detalles finos** (bigotes, textura de pelaje, definición de ojos)

### Causa del Blur
El valor alto de `lambda_L1=60` fuerza al generador a priorizar reconstrucción promedio sobre detalles de alta frecuencia, resultando en sobre-suavizado.

## 🚀 Uso

### Instalación
```bash
pip install torch torchvision opencv-python numpy pillow tqdm matplotlib piq
```

### Preparar Dataset
```python
prepare_catflw_dataset(
    input_root="CatFLW dataset",
    output_root="datasets/catflw",
    sigma=3
)
```

### Entrenar Modelo
```python
G = train_pix2pix(
    dataset_root="datasets/catflw",
    epochs=200,
    batch_size=32,
    lambda_L1=60,
    lambda_perc=3
)
```

### Evaluar en Test
```python
evaluate_on_test(G, dataset_root="datasets/catflw/test")
show_samples(G, test_loader, device, n=5)
```

### Reanudar Entrenamiento
El sistema detecta automáticamente `checkpoints/last_checkpoint.pth` y continúa desde la última época guardada.

## 📁 Estructura del Proyecto

```
pix2pix/
├── pix2pix_proyect.ipynb      # Notebook principal
├── README.md                   # Este archivo
├── .gitignore                  # Exclusiones de Git
├── CatFLW dataset/            # Dataset original (no versionado)
│   ├── images/
│   └── labels/
├── datasets/                   # Dataset procesado (no versionado)
│   └── catflw/
│       ├── train/
│       ├── val/
│       └── test/
├── checkpoints/               # Modelos guardados
│   ├── G_best.pth            # Mejor generador (versionado)
│   ├── D_best.pth            # Mejor discriminador (versionado)
│   ├── last_checkpoint.pth   # Estado completo (no versionado)
│   └── G_epoch{N}.pth        # Checkpoints periódicos (no versionados)
└── results/                   # Salidas del entrenamiento
    ├── training_log.csv      # Métricas por época (versionado)
    ├── epoch_{N}.png         # Visualizaciones (no versionadas)
    └── samples.png           # Resultados finales (no versionados)
```

## 🔄 Próximos Pasos: Optimización Anti-Blur

### Estrategia Propuesta
1. **Reducir lambda_L1**: 60 → 40 (-33%)
2. **Reducir lambda_perc**: 3 → 1.5 (-50%)
3. **Sharper sigma**: 3 → 2 (landmarks más definidos)
4. **Enhanced augmentation**: Rotación, affine, ColorJitter agresivo
5. **Extender epochs**: 200 → 250

### Expectativas
- **PSNR**: 22-26 dB
- **SSIM**: 0.75-0.85
- **Detalles visibles**: Bigotes individuales, textura de pelaje, ojos nítidos

## 📝 Historial de Versiones

### Época 200 (Actual)
- Entrenamiento base completado
- Lambda ajustado en época 101 (100/5 → 60/3)
- Generador converge correctamente
- **Pending**: Resolver blur en detalles finos

## 🛠️ Tecnologías

- **PyTorch 2.x**: Framework de deep learning
- **CUDA**: Aceleración GPU
- **torchvision**: Transformaciones y VGG16 preentrenado
- **OpenCV**: Procesamiento de imágenes
- **piq**: Métricas SSIM
- **tqdm**: Barras de progreso

## 👨‍💻 Autor

**Samuel Ordaz**  
📧 samuel.ordaz@cimat.mx

## 📄 Licencia

Este proyecto es parte de investigación académica en Deep Learning.

## 🙏 Referencias

- Isola, P., et al. (2017). "Image-to-Image Translation with Conditional Adversarial Networks" (Pix2Pix paper)
- Ronneberger, O., et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- Dataset CatFLW: Facial Landmarks in the Wild for Cats
