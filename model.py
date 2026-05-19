"""
モデル定義: ResNet-50 (FractalDB 事前学習) + 埋め込みヘッド + ArcFace
"""
import math

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import ResNet50


class ArcFaceLayer(layers.Layer):
    """ArcFace head.

    入力: 埋め込みベクトル (B, D) と ラベル (B,)
    出力: スケール済みロジット (B, num_classes)
    学習時のみラベルを使ってターゲットクラスに角度マージンを加える。
    """

    def __init__(self, num_classes: int, scale: float = 30.0, margin: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        # 数値安定化のための定数
        self._cos_m = math.cos(margin)
        self._sin_m = math.sin(margin)
        self._threshold = math.cos(math.pi - margin)
        self._mm = self._sin_m * margin  # Easy-margin の境界線形化用

    def build(self, input_shape):
        embed_dim = int(input_shape[-1])
        self.W = self.add_weight(
            name="arcface_W",
            shape=(embed_dim, self.num_classes),
            initializer=tf.keras.initializers.GlorotUniform(),
            regularizer=tf.keras.regularizers.l2(5e-4),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, embeddings, labels=None, training=False):
        x = tf.nn.l2_normalize(embeddings, axis=1)
        w = tf.nn.l2_normalize(self.W, axis=0)
        cos_theta = tf.matmul(x, w)
        cos_theta = tf.clip_by_value(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

        # 推論時 / ラベルがない場合は単純な cos 類似度ロジット
        if labels is None or not training:
            return self.scale * cos_theta

        # cos(theta + m) = cos*cos_m - sin*sin_m
        sin_theta = tf.sqrt(1.0 - tf.square(cos_theta) + 1e-7)
        cos_theta_m = cos_theta * self._cos_m - sin_theta * self._sin_m

        # 数値安定化: theta + m > pi の領域を線形化
        cos_theta_m = tf.where(
            cos_theta > self._threshold,
            cos_theta_m,
            cos_theta - self._mm,
        )

        labels_onehot = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes)
        logits = labels_onehot * cos_theta_m + (1.0 - labels_onehot) * cos_theta
        return self.scale * logits

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_classes": self.num_classes, "scale": self.scale, "margin": self.margin})
        return cfg


def build_models(
    num_classes: int,
    embedding_dim: int = 512,
    input_shape=(224, 224, 3),
    arcface_scale: float = 30.0,
    arcface_margin: float = 0.5,
    dropout: float = 0.4,
):
    """学習用モデル (image + label → logits) と
    推論用モデル (image → embedding) の両方を返す。
    バックボーン重みは学習スクリプト側で別途ロードする。
    """
    image_in = layers.Input(shape=input_shape, name="image")
    label_in = layers.Input(shape=(), dtype=tf.int32, name="label")

    backbone = ResNet50(weights=None, include_top=False, input_shape=input_shape, name="resnet50")
    feat = backbone(image_in)
    feat = layers.GlobalAveragePooling2D(name="gap")(feat)
    feat = layers.Dropout(dropout, name="head_dropout")(feat)
    feat = layers.Dense(embedding_dim, use_bias=False, name="embedding_dense")(feat)
    embedding = layers.BatchNormalization(name="embedding_bn")(feat)

    arcface = ArcFaceLayer(
        num_classes=num_classes,
        scale=arcface_scale,
        margin=arcface_margin,
        name="arcface",
    )
    logits = arcface(embedding, labels=label_in, training=True)

    train_model = Model(inputs=[image_in, label_in], outputs=logits, name="train_model")
    embedding_model = Model(inputs=image_in, outputs=embedding, name="embedding_model")

    return train_model, embedding_model, backbone
