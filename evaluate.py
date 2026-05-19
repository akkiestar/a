"""
評価スクリプト

例:
    python evaluate.py \
        --test_csv data/test.csv \
        --weights ./output/best_embedding.weights.h5 \
        --prototypes ./output/prototypes.npy \
        --num_classes 25
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from data import build_dataset
from model import build_models


def main(args):
    # 埋め込みモデルを構築してロード（ArcFace ヘッドは推論で不要だが、build_models は両方返す）
    _, embedding_model, _ = build_models(
        num_classes=args.num_classes,
        embedding_dim=args.embedding_dim,
        input_shape=(args.image_size, args.image_size, 3),
    )
    embedding_model.load_weights(args.weights)
    print(f"Loaded embedding model: {args.weights}")

    prototypes = np.load(args.prototypes).astype(np.float32)
    print(f"Loaded prototypes: shape={prototypes.shape}")
    assert prototypes.shape[0] == args.num_classes, "プロトタイプ数とクラス数が一致しません"

    test_ds, n_test = build_dataset(
        args.test_csv,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
    )

    confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    sim_top1 = []
    sim_margin = []  # top1 と top2 の差（信頼度の指標）
    correct = 0
    total = 0
    unknown_count = 0

    for images, labels in test_ds:
        emb = embedding_model(images, training=False)
        emb = tf.nn.l2_normalize(emb, axis=1).numpy()
        sims = emb @ prototypes.T  # (B, C)

        # しきい値判定（オープンセット）
        top1 = sims.max(axis=1)
        sorted_sims = np.sort(sims, axis=1)
        top2_margin = sorted_sims[:, -1] - sorted_sims[:, -2]

        preds = sims.argmax(axis=1)
        # しきい値未満は -1（未分類）扱い。混同行列には含めるが accuracy 計算では別カウント
        is_unknown = top1 < args.unknown_threshold

        labels_np = labels.numpy()
        for t, p, unk in zip(labels_np, preds, is_unknown):
            if unk:
                unknown_count += 1
            else:
                confusion[t, p] += 1
                if t == p:
                    correct += 1
            total += 1

        sim_top1.extend(top1.tolist())
        sim_margin.extend(top2_margin.tolist())

    classified = total - unknown_count
    print(f"\n=== Results ===")
    print(f"Total samples       : {total}")
    print(f"Classified          : {classified}")
    print(f"Unknown (< thresh)  : {unknown_count}  (threshold={args.unknown_threshold})")
    if classified > 0:
        print(f"Accuracy (classified): {correct/classified:.4f}  ({correct}/{classified})")
    print(f"Accuracy (all incl. unknown wrong): {correct/total:.4f}")
    print(f"Mean top1 similarity      : {np.mean(sim_top1):.4f}")
    print(f"Mean top1-top2 margin     : {np.mean(sim_margin):.4f}")

    # クラスごとの精度
    print("\nPer-class accuracy:")
    for c in range(args.num_classes):
        n_c = confusion[c].sum()
        if n_c > 0:
            print(f"  class {c:2d}: {confusion[c,c]/n_c:.4f}  ({confusion[c,c]}/{n_c})")

    out_dir = Path(args.weights).parent
    np.save(out_dir / "confusion_matrix.npy", confusion)
    print(f"\nSaved confusion matrix: {out_dir / 'confusion_matrix.npy'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--prototypes", required=True)
    parser.add_argument("--num_classes", type=int, default=25)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--unknown_threshold",
        type=float,
        default=0.0,
        help="top1 cos 類似度がこの値未満は未分類扱い。0.0 なら無効化",
    )
    args = parser.parse_args()
    main(args)
