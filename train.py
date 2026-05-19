"""
学習スクリプト

例:
    python train.py \
        --train_csv data/train.csv \
        --val_csv   data/val.csv \
        --pretrained_weights fractaldb10k_resnet50.weights.h5 \
        --num_classes 25 \
        --output_dir ./output \
        --epochs 50 \
        --batch_size 64
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from data import build_dataset
from model import build_models


def cosine_lr_with_warmup(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * progress))


def compute_embeddings(embedding_model, csv_path, image_size, batch_size):
    ds, _ = build_dataset(
        csv_path, image_size=image_size, batch_size=batch_size, shuffle=False, augment=False
    )
    embs, lbls = [], []
    for images, labels in ds:
        emb = embedding_model(images, training=False)
        emb = tf.nn.l2_normalize(emb, axis=1)
        embs.append(emb.numpy())
        lbls.append(labels.numpy())
    return np.concatenate(embs), np.concatenate(lbls)


def compute_prototypes(embs: np.ndarray, lbls: np.ndarray, num_classes: int) -> np.ndarray:
    dim = embs.shape[1]
    prototypes = np.zeros((num_classes, dim), dtype=np.float32)
    for c in range(num_classes):
        mask = lbls == c
        if mask.sum() == 0:
            continue
        proto = embs[mask].mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        prototypes[c] = proto
    return prototypes


def knn_accuracy(embedding_model, train_csv, val_csv, num_classes, image_size, batch_size):
    train_emb, train_lbl = compute_embeddings(embedding_model, train_csv, image_size, batch_size)
    val_emb, val_lbl = compute_embeddings(embedding_model, val_csv, image_size, batch_size)
    protos = compute_prototypes(train_emb, train_lbl, num_classes)
    sims = val_emb @ protos.T
    preds = sims.argmax(axis=1)
    return float((preds == val_lbl).mean()), protos


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # データ
    train_ds, n_train = build_dataset(
        args.train_csv,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=True,
        augment=True,
    )
    steps_per_epoch = (n_train + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    print(f"Train samples: {n_train} | steps/epoch: {steps_per_epoch} | total steps: {total_steps}")

    # モデル
    train_model, embedding_model, backbone = build_models(
        num_classes=args.num_classes,
        embedding_dim=args.embedding_dim,
        input_shape=(args.image_size, args.image_size, 3),
        arcface_scale=args.arcface_scale,
        arcface_margin=args.arcface_margin,
        dropout=args.dropout,
    )

    if args.pretrained_weights:
        backbone.load_weights(args.pretrained_weights)
        print(f"Loaded pretrained backbone: {args.pretrained_weights}")
    else:
        print("WARNING: No pretrained weights given. Training from scratch.")

    # LP-FT: 最初 warmup_epochs はバックボーン凍結 → 解凍
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.base_lr, momentum=0.9, nesterov=True)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_loss = tf.keras.metrics.Mean()
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = train_model([images, labels], training=True)
            loss = loss_fn(labels, logits)
        grads = tape.gradient(loss, train_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, train_model.trainable_variables))
        train_loss.update_state(loss)
        train_acc.update_state(labels, logits)
        return loss

    history = []
    best_val_acc = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        # LP-FT: 最初の lp_epochs はバックボーン凍結
        if epoch < args.lp_epochs:
            backbone.trainable = False
            phase = "LP"
        else:
            backbone.trainable = True
            phase = "FT"

        train_loss.reset_state()
        train_acc.reset_state()

        for images, labels in train_ds:
            lr = cosine_lr_with_warmup(global_step, total_steps, warmup_steps, args.base_lr)
            optimizer.learning_rate.assign(lr)
            train_step(images, labels)
            global_step += 1

        # 検証 (プロトタイプ k-NN)
        val_acc, _ = knn_accuracy(
            embedding_model,
            args.train_csv,
            args.val_csv,
            args.num_classes,
            args.image_size,
            args.batch_size,
        )

        log = {
            "epoch": epoch + 1,
            "phase": phase,
            "loss": float(train_loss.result()),
            "train_arcface_acc": float(train_acc.result()),
            "val_knn_acc": val_acc,
            "lr": float(optimizer.learning_rate.numpy()),
        }
        history.append(log)
        print(
            f"[{phase}] Epoch {epoch+1}/{args.epochs} | loss={log['loss']:.4f} | "
            f"train_acc={log['train_arcface_acc']:.4f} | val_knn={val_acc:.4f} | lr={log['lr']:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            embedding_model.save_weights(output_dir / "best_embedding.weights.h5")
            print(f"  -> Saved best (val_knn={val_acc:.4f})")

    # 最終: best 重みでプロトタイプを再計算して保存
    embedding_model.load_weights(output_dir / "best_embedding.weights.h5")
    train_emb, train_lbl = compute_embeddings(
        embedding_model, args.train_csv, args.image_size, args.batch_size
    )
    protos = compute_prototypes(train_emb, train_lbl, args.num_classes)
    np.save(output_dir / "prototypes.npy", protos)
    print(f"Saved prototypes: {output_dir / 'prototypes.npy'} (shape={protos.shape})")

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best val_knn_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--pretrained_weights", default=None, help="convert_fractaldb_to_tf.py で生成した .weights.h5")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--num_classes", type=int, default=25)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lp_epochs", type=int, default=3, help="最初の N エポックはバックボーン凍結 (LP-FT)")
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--arcface_scale", type=float, default=30.0)
    parser.add_argument("--arcface_margin", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
