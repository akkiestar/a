"""
FractalDB の PyTorch 事前学習済み ResNet-50 重みを、
tf.keras.applications.ResNet50 互換の TF 重みに変換する。

使い方:
    python convert_fractaldb_to_tf.py \
        --pt_weights FractalDB-10000_res50.pth \
        --tf_output  fractaldb10k_resnet50.weights.h5
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from tensorflow.keras.applications import ResNet50


def load_state_dict(pt_path: str) -> dict:
    """torchvision 互換の state_dict を取り出す（DataParallel / checkpoint ラップに対応）。"""
    try:
        obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(pt_path, map_location="cpu")

    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()

    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict", "net"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

    sd = {k.replace("module.", ""): v for k, v in obj.items()}
    print(f"Loaded PyTorch state_dict with {len(sd)} tensors")
    return sd


def pt_conv_kernel(state_dict: dict, name: str) -> np.ndarray:
    """PyTorch [out,in,H,W] → TF [H,W,in,out]"""
    return state_dict[f"{name}.weight"].numpy().transpose(2, 3, 1, 0)


def pt_bn_weights(state_dict: dict, name: str) -> list:
    """[gamma, beta, moving_mean, moving_variance] の順で返す。"""
    return [
        state_dict[f"{name}.weight"].numpy(),
        state_dict[f"{name}.bias"].numpy(),
        state_dict[f"{name}.running_mean"].numpy(),
        state_dict[f"{name}.running_var"].numpy(),
    ]


def set_conv_weights(tf_model, layer_name: str, state_dict: dict, pt_name: str):
    """conv層の重みを設定。TF側のuse_bias有無を自動判定する。
    PyTorch側のBN付きconvはbiasを持たないので、TF側がbias付きならゼロで埋める。
    """
    layer = tf_model.get_layer(layer_name)
    kernel = pt_conv_kernel(state_dict, pt_name)
    if len(layer.weights) == 2:
        bias_key = f"{pt_name}.bias"
        if bias_key in state_dict:
            bias = state_dict[bias_key].numpy()
        else:
            bias = np.zeros(kernel.shape[-1], dtype=np.float32)
        layer.set_weights([kernel, bias])
    else:
        layer.set_weights([kernel])


def convert(pt_path: str, tf_output: str):
    state_dict = load_state_dict(pt_path)

    tf_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))

    # ステム
    set_conv_weights(tf_model, "conv1_conv", state_dict, "conv1")
    tf_model.get_layer("conv1_bn").set_weights(pt_bn_weights(state_dict, "bn1"))

    block_counts = {1: 3, 2: 4, 3: 6, 4: 3}

    for layer_idx, num_blocks in block_counts.items():
        tf_stage = layer_idx + 1
        for block_idx in range(num_blocks):
            tf_block = block_idx + 1
            pt_pref = f"layer{layer_idx}.{block_idx}"
            tf_pref = f"conv{tf_stage}_block{tf_block}"

            for i, conv_idx in enumerate([1, 2, 3], start=1):
                set_conv_weights(
                    tf_model, f"{tf_pref}_{i}_conv", state_dict, f"{pt_pref}.conv{conv_idx}"
                )
                tf_model.get_layer(f"{tf_pref}_{i}_bn").set_weights(
                    pt_bn_weights(state_dict, f"{pt_pref}.bn{conv_idx}")
                )

            if block_idx == 0:
                set_conv_weights(
                    tf_model, f"{tf_pref}_0_conv", state_dict, f"{pt_pref}.downsample.0"
                )
                tf_model.get_layer(f"{tf_pref}_0_bn").set_weights(
                    pt_bn_weights(state_dict, f"{pt_pref}.downsample.1")
                )

    Path(tf_output).parent.mkdir(parents=True, exist_ok=True)
    tf_model.save_weights(tf_output)
    print(f"Saved TF weights to: {tf_output}")

    dummy = tf.random.normal([1, 224, 224, 3])
    out = tf_model(dummy, training=False)
    print(f"Output shape (sanity check): {out.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_weights", required=True)
    parser.add_argument("--tf_output", required=True)
    args = parser.parse_args()
    convert(args.pt_weights, args.tf_output)
