"""
FractalDB の PyTorch 事前学習済み ResNet-50 重みを、
tf.keras.applications.ResNet50 互換の TF 重みに変換する。

使い方:
    python convert_fractaldb_to_tf.py \
        --pt_weights FractalDB-10000_resnet50_epoch90.pth \
        --tf_output  fractaldb10k_resnet50.weights.h5

依存:
    torch, tensorflow>=2.10
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from tensorflow.keras.applications import ResNet50


def load_state_dict(pt_path: str) -> dict:
    """torchvision 互換の state_dict を取り出す（DataParallel / checkpoint ラップに対応）。"""
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    # 'module.' プレフィクスを削除
    return {k.replace("module.", ""): v for k, v in obj.items()}


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


def convert(pt_path: str, tf_output: str):
    state_dict = load_state_dict(pt_path)
    print(f"Loaded PyTorch state_dict with {len(state_dict)} tensors")

    # TF ResNet-50 を重みなしで構築（バックボーンのみ）
    tf_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))

    # ステム
    tf_model.get_layer("conv1_conv").set_weights([pt_conv_kernel(state_dict, "conv1")])
    tf_model.get_layer("conv1_bn").set_weights(pt_bn_weights(state_dict, "bn1"))

    # 各ステージのブロック数（torchvision の layer1〜4）
    block_counts = {1: 3, 2: 4, 3: 6, 4: 3}

    for layer_idx, num_blocks in block_counts.items():
        tf_stage = layer_idx + 1  # layer1 → conv2_block*, layer4 → conv5_block*
        for block_idx in range(num_blocks):
            tf_block = block_idx + 1
            pt_pref = f"layer{layer_idx}.{block_idx}"
            tf_pref = f"conv{tf_stage}_block{tf_block}"

            # 3つの conv-bn (1x1, 3x3, 1x1)
            for i, conv_idx in enumerate([1, 2, 3], start=1):
                tf_model.get_layer(f"{tf_pref}_{i}_conv").set_weights(
                    [pt_conv_kernel(state_dict, f"{pt_pref}.conv{conv_idx}")]
                )
                tf_model.get_layer(f"{tf_pref}_{i}_bn").set_weights(
                    pt_bn_weights(state_dict, f"{pt_pref}.bn{conv_idx}")
                )

            # ショートカット（各ステージの先頭ブロックのみ）
            if block_idx == 0:
                tf_model.get_layer(f"{tf_pref}_0_conv").set_weights(
                    [pt_conv_kernel(state_dict, f"{pt_pref}.downsample.0")]
                )
                tf_model.get_layer(f"{tf_pref}_0_bn").set_weights(
                    pt_bn_weights(state_dict, f"{pt_pref}.downsample.1")
                )

    Path(tf_output).parent.mkdir(parents=True, exist_ok=True)
    tf_model.save_weights(tf_output)
    print(f"Saved TF weights to: {tf_output}")

    # 簡易動作確認
    dummy = tf.random.normal([1, 224, 224, 3])
    out = tf_model(dummy, training=False)
    print(f"Output shape (sanity check): {out.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_weights", required=True, help="FractalDB の .pth ファイルパス")
    parser.add_argument("--tf_output", required=True, help="変換後の TF 重み保存先 (.weights.h5)")
    args = parser.parse_args()
    convert(args.pt_weights, args.tf_output)
