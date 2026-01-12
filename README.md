# CNN Mechanistic Analysis (Grad-CAM + Intermediate Activations)

Single-file tooling for mechanistic probing of CNN image classifiers:
- **Grad-CAM** heatmaps (class-specific evidence localization)
- **Intermediate feature-map grids** for selected layers

The script runs on ImageNet-pretrained Keras models by default (VGG16/ResNet50/EfficientNetB0/MobileNetV2), and can optionally load custom weights if they match the chosen backbone.

---

## Installation

```bash
pip install tensorflow opencv-python matplotlib numpy
```

## Quickstart

```
python cnn_mechanistic_analysis.py \
  --image /path/to/image.jpg \
  --model vgg16 \
  --outdir outputs
```
This writes:
* outputs/input.png
* outputs/gradcam_heatmap.png
* outputs/gradcam_overlay.png
and prints the top-k decoded predictions to stdout.

## Visualize intermediate activations (feature maps)

Pass comma-separated layer names:
```
python cnn_mechanistic_analysis_clean.py \
  --image /path/to/image.jpg \
  --model vgg16 \
  --activation-layers block3_conv3,block4_conv3,block5_conv3 \
  --outdir outputs
```

This additionally writes:
* outputs/features_block3_conv3.png
* outputs/features_block4_conv3.png
* outputs/features_block5_conv3.png

You can control channel tiling with:
```
--max-channels 64
```

## Notes on layer selection
# Grad-CAM layer
By default, the script uses a backbone-specific last convolution layer:
* vgg16: block5_conv3
* resnet50: conv5_block3_out
* efficientnetb0: top_conv
* mobilenetv2: Conv_1
  
You can override this with:
```
--last-conv <layer_name>
```
# Finding valid layer names
```
import tensorflow as tf
from tensorflow.keras.applications import vgg16
m = vgg16.VGG16(weights="imagenet", include_top=True)
print([l.name for l in m.layers])
```
## Using custom weights (optional)
If you trained/fine-tuned a model that matches the chosen backbone:
```
python cnn_mechanistic_analysis_clean.py \
  --image /path/to/image.jpg \
  --model resnet50 \
  --weights /path/to/weights.h5 \
  --outdir outputs
```
## Reproducibility

The script sets NumPy + TensorFlow seeds (default 1234). Override via:
```
--seed 42
```
## Outputs and interpretation
Grad-CAM overlay helps localize which regions contributed most to the predicted class.
Feature-map grids help inspect internal representations (e.g., overly localized texture detectors vs broader structure sensitivity), and are often more informative in weak-signal regimes.
