"""
CSV (image_path, label) → tf.data.Dataset

CSV のカラム名はデフォルトで "image_path" / "label"。
画像は JPEG / PNG どちらにも対応。
"""
from typing import Tuple

import pandas as pd
import tensorflow as tf

# ImageNet 互換の正規化（FractalDB 事前学習はファインチューニング時に ImageNet stats を使うのが一般的）
_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def _decode_image(path: tf.Tensor, image_size: int) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [image_size, image_size], method="bilinear")
    return img


def _augment(img: tf.Tensor) -> tf.Tensor:
    """印字欠陥タスク向けの augmentation。
    水平反転は文字の意味を壊すため使わない。袋の歪み・撮影条件の揺れを再現する。
    """
    img = tf.image.random_brightness(img, max_delta=20.0)  # 0-255 スケールでの ±20
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    img = tf.image.random_saturation(img, lower=0.9, upper=1.1)

    # 軽い回転（±10度）と平行移動
    angle = tf.random.uniform([], -10.0 * 3.14159 / 180.0, 10.0 * 3.14159 / 180.0)
    img = _rotate(img, angle)

    img = tf.clip_by_value(img, 0.0, 255.0)
    return img


def _rotate(img: tf.Tensor, angle: tf.Tensor) -> tf.Tensor:
    """単純な回転（バイリニア補間）。"""
    img = tf.expand_dims(img, 0)
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    # [a0, a1, a2, b0, b1, b2, c0, c1]
    transform = tf.stack(
        [cos_a, -sin_a, 0.0, sin_a, cos_a, 0.0, 0.0, 0.0]
    )
    transform = tf.expand_dims(transform, 0)
    h = tf.cast(tf.shape(img)[1], tf.float32)
    w = tf.cast(tf.shape(img)[2], tf.float32)
    # 中心回転に補正
    cx, cy = w / 2.0, h / 2.0
    tx = cx - cos_a * cx + sin_a * cy
    ty = cy - sin_a * cx - cos_a * cy
    transform = tf.stack([[cos_a, -sin_a, tx, sin_a, cos_a, ty, 0.0, 0.0]])
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img,
        transforms=transform,
        output_shape=tf.shape(img)[1:3],
        fill_value=0.0,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
    )
    return tf.squeeze(img, 0)


def _normalize(img: tf.Tensor) -> tf.Tensor:
    img = tf.cast(img, tf.float32) / 255.0
    return (img - _MEAN) / _STD


def build_dataset(
    csv_path: str,
    image_size: int = 224,
    batch_size: int = 64,
    shuffle: bool = True,
    augment: bool = True,
    path_col: str = "image_path",
    label_col: str = "label",
) -> Tuple[tf.data.Dataset, int]:
    """CSV からデータセットを構築する。

    Returns:
        (tf.data.Dataset, サンプル数)
        Dataset は (image, label) を yield する。
    """
    df = pd.read_csv(csv_path)
    paths = df[path_col].astype(str).values
    labels = df[label_col].astype("int32").values
    n = len(paths)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(n, 10000), reshuffle_each_iteration=True)

    def _load(path, label):
        img = _decode_image(path, image_size)
        if augment:
            img = _augment(img)
        img = _normalize(img)
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, n
