# 🎨 Guía Completa: Pix2Pix para Generación de Imágenes desde Landmarks

## 📚 Tabla de Contenidos

1. [Introducción](#introducción)
2. [Conceptos Fundamentales](#conceptos-fundamentales)
3. [Arquitectura del Modelo](#arquitectura-del-modelo)
4. [Preparación del Dataset](#preparación-del-dataset)
5. [Entrenamiento](#entrenamiento)
6. [Evaluación y Métricas](#evaluación-y-métricas)
7. [Flujo Completo del Código](#flujo-completo-del-código)

---

## 🎯 Introducción

### ¿Qué es Pix2Pix?

**Pix2Pix** es una arquitectura de **GAN condicional (cGAN)** que aprende a traducir imágenes de un dominio a otro:
- **Input**: Imagen condicional (landmarks/máscara)
- **Output**: Imagen realista generada

**Aplicaciones**:
- Boceto → Foto realista
- Mapa → Imagen satelital
- Día → Noche
- **Nuestro caso**: Landmarks faciales → Cara de gato

---

## 🧠 Conceptos Fundamentales

### 1. **Redes Generativas Adversarias (GANs)**

Dos redes que compiten entre sí:

```
┌─────────────┐         ┌──────────────┐
│  Generador  │ ──────→ │Discriminador │
│     (G)     │  Fake   │     (D)      │
└─────────────┘         └──────────────┘
      ↑                        │
      │                        │
      └────── Feedback ────────┘
```

- **Generador (G)**: Crea imágenes falsas intentando engañar al discriminador
- **Discriminador (D)**: Clasifica si una imagen es real o falsa

**Objetivo**: G mejora generando imágenes más realistas, D mejora detectando falsas.

---

### 2. **Conditional GAN (cGAN)**

A diferencia de GANs clásicas, **Pix2Pix es condicional**:

```
Input (Landmarks) + Ruido → Generador → Output (Cara)
                              ↓
            Discriminador compara: Input + Output vs Input + Real
```

**Ventaja**: Control total sobre la salida (condicionada al input).

---

### 3. **U-Net Generator**

Arquitectura **encoder-decoder** con **skip connections**:

```
Input (256×256)
    ↓
┌─────────────────────────────────────┐
│  Encoder (Downsampling)             │
│  64 → 128 → 256 → 512 → 512 (×3)    │  Extrae características
└─────────────────────────────────────┘
         ↓ Bottleneck (512)
┌─────────────────────────────────────┐
│  Decoder (Upsampling)               │
│  512 → 512 (×3) → 256 → 128 → 64    │  Reconstruye imagen
└─────────────────────────────────────┘
    ↓
Output (256×256×3)
```

**Skip connections** (concatenaciones):
- Conectan capas del encoder directamente al decoder
- Preservan detalles finos (bordes, texturas)
- Evitan pérdida de información espacial

#### Código U-Net:

```python
# Encoder
d1 = self.down1(x)        # 256→128 (64 channels)
d2 = self.down2(d1)       # 128→64  (128 channels)
# ... más capas

# Decoder con skip connections
u1 = self.up1(bottleneck)
u2 = self.up2(torch.cat([u1, d7], 1))  # ← Skip connection
```

---

### 4. **PatchGAN Discriminator**

No clasifica la imagen completa, sino **parches pequeños** (30×30):

```
Imagen 256×256 → Convs → Salida 30×30×1
                          ↓
                Cada píxel = probabilidad de ese parche ser real
```

**Ventajas**:
- Menos parámetros que un discriminador global
- Enfoca en detalles locales (texturas)
- Mejor para imágenes de alta resolución

#### Arquitectura:

```python
Conv(4, 64)  → LeakyReLU
Conv(64, 128) + BN → LeakyReLU
Conv(128, 256) + BN → LeakyReLU
Conv(256, 512) + BN → LeakyReLU
Conv(512, 1)  # Sin activación (usa BCEWithLogitsLoss)
```

---

## 📦 Preparación del Dataset

### 1. **Mapas Gaussianos para Landmarks**

En lugar de puntos discretos, creamos **heatmaps suaves**:

```python
def create_heatmap_landmarks(coords_scaled, size=256, sigma=4):
    heatmap = np.zeros((size, size))
    coeff = 1.0 / (2 * sigma**2)
    
    for x, y in coords_scaled:
        # Gaussiana 2D centrada en (x, y)
        dist_sq = (x_grid - x)**2 + (y_grid - y)**2
        heatmap += np.exp(-dist_sq * coeff)
```

**¿Por qué Gaussianas?**
- Más información espacial que puntos binarios
- El generador aprende mejor con gradientes suaves
- Sigma bajo (3-4) = landmarks más definidos

**Visualización**:
```
Punto (x, y)   →   Gaussiana σ=4
     *         →      .:*.:.
                      :*#*:
                      .:*:.
```

---

### 2. **Estructura del Dataset**

```
datasets/catflw/
├── train/
│   ├── A/  ← Mapas Gaussianos (landmarks)
│   └── B/  ← Imágenes reales
├── val/
│   ├── A/
│   └── B/
└── test/
    ├── A/
    └── B/
```

**Split**: 80% train, 10% val, 10% test

---

### 3. **Transformaciones (Data Augmentation)**

```python
# Para A (landmarks)
transforms.Resize((256, 256), interpolation=NEAREST)  # Preserva valores discretos
transforms.RandomHorizontalFlip(p=0.5)
transforms.ToTensor()
transforms.Normalize([0.5], [0.5])  # [-1, 1]

# Para B (imágenes)
transforms.Resize((256, 256), interpolation=BILINEAR)  # Suaviza
transforms.RandomHorizontalFlip(p=0.5)
transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)  # Variación de color
transforms.ToTensor()
transforms.Normalize([0.5]*3, [0.5]*3)  # [-1, 1]
```

---

## 🎓 Entrenamiento

### 1. **Funciones de Pérdida**

#### a) **GAN Loss (Adversarial)**

```python
criterion_GAN = nn.BCEWithLogitsLoss()

# Para Discriminador:
loss_D = 0.5 * (loss_real + loss_fake)

# Para Generador:
loss_GAN = criterion_GAN(pred_fake, ones)  # Quiere engañar a D
```

**Concepto**: G intenta maximizar la probabilidad de que D clasifique sus salidas como reales.

---

#### b) **L1 Loss (Reconstrucción)**

```python
criterion_L1 = nn.L1Loss()
loss_L1 = criterion_L1(fake_B, B) * lambda_L1  # λ=100
```

**¿Por qué L1 y no L2 (MSE)?**
- **L1** penaliza diferencias absolutas → menos blur
- **L2** penaliza diferencias al cuadrado → imágenes más borrosas
- L1 preserva mejor bordes y detalles

**Analogía**:
```
Error = 10 píxeles
L1: penaliza 10
L2: penaliza 100 (10²) → sobreenaltiza errores grandes → promedia colores → blur
```

---

#### c) **Perceptual Loss (VGG)**

```python
vgg = models.vgg16(...).features[:16]  # Capas convolucionales

def perceptual_loss(fake, real):
    return F.l1_loss(vgg(fake), vgg(real))  # Compara features
```

**Concepto**: En lugar de comparar píxel a píxel, compara **características semánticas** extraídas por VGG.

**Ventajas**:
- Mejor percepción humana de similitud
- Preserva texturas y estructuras de alto nivel

---

#### d) **Pérdida Total del Generador**

```python
loss_G = loss_GAN + loss_L1 * 100 + loss_perceptual * 10
         ↑           ↑                ↑
     Engañar D   Parecerse       Texturas/semántica
                 píxel a píxel
```

**Pesos típicos**:
- `lambda_L1 = 100`: Fuerte penalización por diferencias píxel
- `lambda_perc = 10`: Moderado enfoque en features

---

### 2. **Label Smoothing con Ruido**

```python
# En lugar de usar 1.0 y 0.0:
real_label = 0.9 + 0.1 * torch.rand_like(pred_real)  # [0.9, 1.0]
fake_label = 0.1 * torch.rand_like(pred_fake)        # [0.0, 0.1]
```

**Beneficios**:
- Previene overconfidence del discriminador
- Estabiliza entrenamiento GAN
- Reduce mode collapse

---

### 3. **Automatic Mixed Precision (AMP)**

```python
scaler = torch.cuda.amp.GradScaler()

with torch.amp.autocast("cuda"):
    fake_B = G(A)
    loss_G = ...

scaler.scale(loss_G).backward()
scaler.step(opt_G)
scaler.update()
```

**¿Qué hace?**
- Usa **FP16** (16-bit) para forward/backward → **2× más rápido**
- Mantiene **FP32** (32-bit) para actualizaciones de pesos → estabilidad
- `GradScaler` escala gradientes para evitar underflow

**Ventaja**: Entrenar el doble de rápido con misma GPU.

---

### 4. **Learning Rate Scheduling**

```python
scheduler = LambdaLR(opt, lr_lambda=lambda e: 1 - max(0, e - epochs/2) / (epochs/2))
```

**Comportamiento**:
```
Epochs 1-100:    lr = 2e-4 (constante)
Epochs 101-200:  lr decae linealmente → 0
```

**Razón**: Al inicio necesita explorar, al final afinar detalles.

---

### 5. **Checkpoints y Reanudación**

```python
torch.save({
    "epoch": epoch,
    "G_state": G.state_dict(),
    "D_state": D.state_dict(),
    "opt_G_state": opt_G.state_dict(),
    "opt_D_state": opt_D.state_dict(),
    "sched_G_state": scheduler_G.state_dict(),
    "sched_D_state": scheduler_D.state_dict(),
    "scaler_state": scaler.state_dict(),
    "best_val_loss": best_val_loss
}, "checkpoints/last_checkpoint.pth")
```

**Tipos de checkpoint**:
- `last_checkpoint.pth`: Última época (para reanudar)
- `G_best.pth`, `D_best.pth`: Mejor modelo (menor val loss)
- `G_epoch{N}.pth`: Snapshots cada 50 épocas

---

## 📊 Evaluación y Métricas

### 1. **PSNR (Peak Signal-to-Noise Ratio)**

```python
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
```

**Interpretación**:
- Mide similitud píxel a píxel
- **Mayor = mejor** (típicamente 20-40 dB)
- PSNR > 30 dB → buena calidad

---

### 2. **SSIM (Structural Similarity Index)**

```python
import piq
ssim_value = piq.ssim(pred, target, data_range=2.0)
```

**Concepto**: Mide similitud estructural (luminancia, contraste, estructura) en lugar de píxeles brutos.

**Interpretación**:
- Rango: [0, 1]
- **1.0 = idénticas**
- SSIM > 0.9 → muy buena calidad
- Correlaciona mejor con percepción humana que PSNR

---

### 3. **L1 Loss**

```python
criterion_L1(fake_B, B)  # Diferencia promedio absoluta
```

**Interpretación**:
- Menor = más parecido
- Típicamente 0.05-0.20 en imágenes normalizadas

---

## 🔄 Flujo Completo del Código

### **Paso 1: Preparar Dataset**

```python
prepare_catflw_dataset("CatFLW dataset", "datasets/catflw", sigma=3)
```

1. Lee imágenes y JSONs con landmarks
2. Recorta caras usando bounding boxes
3. Escala a 256×256
4. Genera heatmaps Gaussianos para landmarks
5. Guarda pares A (landmarks) y B (imágenes)

---

### **Paso 2: Crear DataLoaders**

```python
train_dataset = Pix2PixDataset(root, augment=True)
train_loader = DataLoader(train_dataset, batch_size=32, ...)
```

- Carga pares (A, B)
- Aplica augmentation (flips, color jitter)
- Normaliza a [-1, 1]

---

### **Paso 3: Inicializar Modelos**

```python
G = UNetGenerator(in_ch=1, out_ch=3)  # 1 canal → 3 canales RGB
D = PatchDiscriminator(in_ch=4)       # 1 (A) + 3 (B) = 4 canales
```

---

### **Paso 4: Loop de Entrenamiento**

```python
for epoch in range(epochs):
    for A, B in train_loader:
        # 1) Entrenar Discriminador
        fake_B = G(A).detach()
        loss_D = BCE(D(A, B), real) + BCE(D(A, fake_B), fake)
        
        # 2) Entrenar Generador
        fake_B = G(A)
        loss_G = BCE(D(A, fake_B), real) + L1(fake_B, B) + Perc(fake_B, B)
        
    # 3) Validación
    val_loss = evaluate(val_loader)
    
    # 4) Guardar checkpoints
    save_checkpoint(...)
```

---

### **Paso 5: Evaluación**

```python
evaluate_on_test(G, dataset_root="datasets/catflw/test")
```

- Calcula L1, PSNR, SSIM en conjunto de test
- No afecta el entrenamiento (solo diagnosis)

---

### **Paso 6: Visualización**

```python
show_samples(G, test_loader, device, n=5)
```

- Muestra comparación lado a lado:
  ```
  Input A (Landmarks) | Real B (Target) | Generado
  ```

---

## 💡 Conceptos Avanzados

### 1. **¿Por qué U-Net?**

Otras opciones:
- **Encoder-Decoder simple**: Pierde detalles espaciales en el bottleneck
- **ResNet**: No diseñado para imagen a imagen
- **U-Net**: Skip connections preservan información de cada escala

---

### 2. **¿Por qué PatchGAN?**

Alternativas:
- **Discriminador global**: Clasifica imagen completa → no captura texturas locales
- **PatchGAN**: Cada parche se evalúa independientemente → mejor detalle

---

### 3. **Orden de Entrenamiento: D luego G**

```python
# Primero D
opt_D.zero_grad()
loss_D.backward()
opt_D.step()

# Luego G
opt_G.zero_grad()
loss_G.backward()
opt_G.step()
```

**Razón**: D debe estar actualizado para dar feedback correcto a G.

---

### 4. **`.detach()` en fake_B**

```python
fake_B = G(A).detach()  # Para entrenar D
```

**¿Por qué?**
- Cuando entrenamos D, NO queremos actualizar G
- `.detach()` rompe el grafo computacional hacia G
- Sin esto, `loss_D.backward()` actualizaría también G

---

### 5. **BatchNorm vs sin Bias**

```python
nn.Conv2d(..., bias=False)
nn.BatchNorm2d(...)
```

**Razón**: BatchNorm normaliza y añade parámetros aprendibles (γ, β), haciendo el bias redundante.

---

## 📈 Hiperparámetros Clave

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `batch_size` | 32 | Balance memoria/velocidad |
| `lr` | 2e-4 | Estándar GANs (Adam) |
| `beta1` | 0.5 | Momentum bajo para GANs |
| `lambda_L1` | 100 | Fuerte reconstrucción |
| `lambda_perc` | 10 | Moderado perceptual |
| `sigma` | 3-4 | Landmarks definidos |
| `epochs` | 100-200 | Convergencia típica |

---

## 🛠️ Troubleshooting

### **Problema**: Mode Collapse
- **Síntoma**: G genera siempre la misma imagen
- **Solución**: 
  - Reducir lr de G
  - Aumentar label smoothing
  - Agregar más augmentation

---

### **Problema**: D demasiado fuerte
- **Síntoma**: loss_D → 0, loss_G no baja
- **Solución**: 
  - Entrenar D cada 2-3 iteraciones
  - Reducir lr de D

---

### **Problema**: Imágenes borrosas
- **Síntoma**: Output realista pero desenfocado
- **Solución**: 
  - Aumentar `lambda_L1`
  - Reducir `lambda_perc`
  - Verificar que usas L1 (no L2)

---

## 🎯 Mejoras Posibles

1. **Spectral Normalization**: Estabiliza D
2. **Self-Attention**: Captura dependencias globales
3. **Progressive Growing**: Entrenar desde 64×64 → 256×256
4. **Gradient Penalty**: Alternativa a label smoothing
5. **Multi-Scale Discriminator**: Evalúa múltiples resoluciones

---

## 📚 Referencias

- [Pix2Pix Paper (2017)](https://arxiv.org/abs/1611.07004)
- [U-Net Architecture](https://arxiv.org/abs/1505.04597)
- [PatchGAN](https://arxiv.org/abs/1611.07004)
- [Perceptual Losses](https://arxiv.org/abs/1603.08155)

---

## ✅ Checklist de Entrenamiento

- [ ] Dataset preparado correctamente (splits 80/10/10)
- [ ] Visualizar ejemplos del dataset (A y B alineados)
- [ ] Verificar GPU disponible (`torch.cuda.is_available()`)
- [ ] Iniciar con pocas épocas (10) para validar
- [ ] Monitorear losses (D y G deben oscilar, no diverger)
- [ ] Guardar checkpoints periódicamente
- [ ] Evaluar en test set al final
- [ ] Visualizar resultados cualitativos

---

## 🚀 Comando de Ejecución

```python
# En Jupyter/VS Code notebook:
# Ejecutar todas las celdas en orden

# Si quieres solo entrenar:
G = train_pix2pix(
    dataset_root="datasets/catflw",
    epochs=100,
    batch_size=32,
    lambda_L1=100,
    lambda_perc=5
)

# Evaluar:
evaluate_on_test(G)
show_samples(G, test_loader, device, n=5)
```

---

**¡Listo!** Ahora tienes una comprensión completa de cómo funciona Pix2Pix y cada parte del código. 🎨✨
