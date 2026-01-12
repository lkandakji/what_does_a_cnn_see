"""
cnn_mechanistic_analysis.py

Mechanistic probes for CNN classifiers:
- Grad-CAM heatmaps (class-specific evidence localization)
- Intermediate feature-map visualization for selected layers

This script is intentionally single-file and self-contained so it can be dropped into
an existing project or run standalone from the command line.

Example:
  python cnn_mechanistic_analysis.py --image /path/to/img.jpg --model vgg16 --outdir outputs

Notes:
- Works with ImageNet-pretrained Keras models by default.
- For custom fine-tuned models, point --weights to a .h5 / SavedModel directory that
  matches the chosen backbone architecture and input size.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

# Optional deps; import with clear error messages.
try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("opencv-python is required: pip install opencv-python") from e

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("matplotlib is required: pip install matplotlib") from e

try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import Model  # type: ignore
    from tensorflow.keras.applications import (
        vgg16,
        resnet50,
        efficientnet,
        mobilenet_v2,
    )
except Exception as e:  # pragma: no cover
    raise ImportError("tensorflow is required: pip install tensorflow") from e


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    build_fn: callable
    preprocess_fn: callable
    decode_fn: callable
    default_input_size: int
    default_last_conv: str


_BACKBONES = {
    "vgg16": BackboneSpec(
        name="vgg16",
        build_fn=lambda input_size: vgg16.VGG16(weights="imagenet", include_top=True, input_shape=(input_size, input_size, 3)),
        preprocess_fn=vgg16.preprocess_input,
        decode_fn=vgg16.decode_predictions,
        default_input_size=224,
        default_last_conv="block5_conv3",
    ),
    "resnet50": BackboneSpec(
        name="resnet50",
        build_fn=lambda input_size: resnet50.ResNet50(weights="imagenet", include_top=True, input_shape=(input_size, input_size, 3)),
        preprocess_fn=resnet50.preprocess_input,
        decode_fn=resnet50.decode_predictions,
        default_input_size=224,
        default_last_conv="conv5_block3_out",
    ),
    "efficientnetb0": BackboneSpec(
        name="efficientnetb0",
        build_fn=lambda input_size: efficientnet.EfficientNetB0(weights="imagenet", include_top=True, input_shape=(input_size, input_size, 3)),
        preprocess_fn=efficientnet.preprocess_input,
        decode_fn=efficientnet.decode_predictions,
        default_input_size=224,
        default_last_conv="top_conv",
    ),
    "mobilenetv2": BackboneSpec(
        name="mobilenetv2",
        build_fn=lambda input_size: mobilenet_v2.MobileNetV2(weights="imagenet", include_top=True, input_shape=(input_size, input_size, 3)),
        preprocess_fn=mobilenet_v2.preprocess_input,
        decode_fn=mobilenet_v2.decode_predictions,
        default_input_size=224,
        default_last_conv="Conv_1",
    ),
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_image_rgb(path: Path, input_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads an image from disk.

    Returns:
      rgb_uint8: HxWx3 uint8 image in RGB (original size)
      rgb_resized: input_size x input_size x 3 uint8 image in RGB
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_AREA)
    return rgb, resized


def preprocess(resized_rgb_uint8: np.ndarray, preprocess_fn) -> tf.Tensor:
    x = resized_rgb_uint8.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_fn(x)
    return tf.convert_to_tensor(x)


def build_model(spec: BackboneSpec, input_size: int, weights: Optional[str]) -> Model:
    model = spec.build_fn(input_size)
    if weights:
        w = Path(weights)
        if not w.exists():
            raise FileNotFoundError(f"--weights path does not exist: {w}")
        model.load_weights(str(w))
    return model


def gradcam_heatmap(
    model: Model,
    image_tensor: tf.Tensor,
    class_index: int,
    last_conv_layer_name: str,
) -> np.ndarray:
    """
    Computes Grad-CAM heatmap for a given class index.

    Returns:
      heatmap: HxW float in [0, 1]
    """
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except Exception as e:
        raise ValueError(
            f"Could not find layer '{last_conv_layer_name}'. "
            f"Available layers include: {[l.name for l in model.layers[:15]]} ..."
        ) from e

    grad_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(image_tensor, training=False)
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_output)
    if grads is None:
        raise RuntimeError("Gradients are None. Check that the chosen layer is differentiable and used by the output.")

    # Global-average pooling over spatial dims
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]  # HxWxC
    heatmap = tf.reduce_sum(tf.multiply(conv_output, pooled_grads), axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    denom = tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap / denom
    return heatmap.numpy()


def overlay_heatmap(
    original_rgb_uint8: np.ndarray,
    heatmap_01: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlays a heatmap (H'xW') onto the original image (HxW), returning RGB uint8.
    """
    h, w = original_rgb_uint8.shape[:2]
    heatmap = cv2.resize(heatmap_01, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_u8 = np.clip(255 * heatmap, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap_color + (1 - alpha) * original_rgb_uint8).astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def get_layer_activations(model: Model, image_tensor: tf.Tensor, layer_names: List[str]) -> List[np.ndarray]:
    """
    Returns activations for the given layer names for a single input image.
    """
    outputs = []
    for name in layer_names:
        try:
            outputs.append(model.get_layer(name).output)
        except Exception as e:
            raise ValueError(f"Layer '{name}' not found in model.") from e
    act_model = Model(inputs=model.inputs, outputs=outputs)
    acts = act_model(image_tensor, training=False)
    # Keras returns a Tensor or list of Tensors depending on n outputs
    if isinstance(acts, (list, tuple)):
        return [a.numpy() for a in acts]
    return [acts.numpy()]


def tile_feature_maps(
    activation: np.ndarray,
    max_channels: int = 64,
    images_per_row: int = 8,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Tiles feature maps from a single conv activation (1xHxWxC) into a grid image (uint8).
    """
    if activation.ndim != 4 or activation.shape[0] != 1:
        raise ValueError(f"Expected activation shape (1,H,W,C). Got {activation.shape}")

    _, h, w, c = activation.shape
    c = min(c, max_channels)
    n_cols = int(np.ceil(c / images_per_row))
    grid = np.zeros((n_cols * h, images_per_row * w), dtype=np.uint8)

    for idx in range(c):
        row = idx // images_per_row
        col = idx % images_per_row
        fm = activation[0, :, :, idx].copy()
        fm = fm - fm.mean()
        fm = fm / (fm.std() + eps)
        fm = fm * 64 + 128  # contrast stretch
        fm = np.clip(fm, 0, 255).astype(np.uint8)
        grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = fm

    return grid


def save_fig_rgb(path: Path, rgb_uint8: np.ndarray, title: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(rgb_uint8)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(path), dpi=200)
    plt.close()


def save_fig_gray(path: Path, gray_uint8: np.ndarray, title: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.imshow(gray_uint8, cmap="gray")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(path), dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mechanistic probes for CNN classifiers (Grad-CAM + feature maps).")
    p.add_argument("--image", type=str, required=True, help="Path to an input image (jpg/png).")
    p.add_argument("--model", type=str, default="vgg16", choices=sorted(_BACKBONES.keys()), help="Backbone architecture.")
    p.add_argument("--weights", type=str, default=None, help="Optional path to custom weights (Keras .h5).")
    p.add_argument("--input-size", type=int, default=None, help="Override input resolution (default: backbone default).")
    p.add_argument("--last-conv", type=str, default=None, help="Layer name for Grad-CAM (default: backbone default).")
    p.add_argument("--topk", type=int, default=5, help="Top-k decoded predictions to print.")
    p.add_argument("--outdir", type=str, default="outputs", help="Directory to write outputs.")
    p.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha for heatmap.")
    p.add_argument(
        "--activation-layers",
        type=str,
        default="",
        help="Comma-separated layer names to visualize feature maps (e.g., block3_conv3,block4_conv3).",
    )
    p.add_argument("--max-channels", type=int, default=64, help="Max feature maps per layer to tile.")
    p.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    spec = _BACKBONES[args.model]
    input_size = args.input_size or spec.default_input_size
    last_conv = args.last_conv or spec.default_last_conv

    image_path = Path(args.image)
    outdir = Path(args.outdir)

    original_rgb, resized_rgb = load_image_rgb(image_path, input_size)
    x = preprocess(resized_rgb, spec.preprocess_fn)

    model = build_model(spec, input_size, args.weights)

    preds = model(x, training=False).numpy()
    topk = spec.decode_fn(preds, top=args.topk)[0]

    print(f"\nModel: {spec.name} | input_size={input_size} | last_conv={last_conv}")
    print("Top predictions:")
    for (_, label, prob) in topk:
        print(f"  {label:>20s}: {prob:.4f}")

    class_index = int(np.argmax(preds[0]))
    heatmap = gradcam_heatmap(model, x, class_index, last_conv)
    overlay = overlay_heatmap(original_rgb, heatmap, alpha=args.alpha)

    save_fig_rgb(outdir / "input.png", original_rgb, title="Input")
    save_fig_rgb(outdir / "gradcam_overlay.png", overlay, title="Grad-CAM overlay")
    # Save heatmap alone (rescaled)
    heatmap_u8 = np.clip(255 * heatmap, 0, 255).astype(np.uint8)
    save_fig_gray(outdir / "gradcam_heatmap.png", heatmap_u8, title="Grad-CAM heatmap")

    if args.activation_layers.strip():
        layer_names = [s.strip() for s in args.activation_layers.split(",") if s.strip()]
        activations = get_layer_activations(model, x, layer_names)
        for name, act in zip(layer_names, activations):
            grid = tile_feature_maps(act, max_channels=args.max_channels)
            save_fig_gray(outdir / f"features_{name}.png", grid, title=f"Feature maps: {name}")

    print(f"\nSaved outputs to: {outdir.resolve()}\n")


if __name__ == "__main__":
    main()
