"""
postprocess.py – 異常マップからの異常座標抽出ユーティリティ

model.py が出力する (B, H, W, 1) のシグモイド異常マップに対し、
閾値以上の連結領域を検出して各領域の中心座標を返す。

使用例
------
    from model import build_siamese_anomaly_detector
    from postprocess import extract_anomaly_centroids_batch

    model = build_siamese_anomaly_detector((256, 256), channels=1)
    anomaly_maps = model.predict([target_batch, ref_batch])  # (B, H, W, 1)

    results = extract_anomaly_centroids_batch(anomaly_maps, threshold=0.5,
                                              min_size=2)
    for i, (centers, sizes) in enumerate(results):
        print(f"image {i}: {len(centers)} anomalies")
        for (y, x), s in zip(centers, sizes):
            print(f"  (y={y:.2f}, x={x:.2f})  size={s}")
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def extract_anomaly_centroids(
    anomaly_map: np.ndarray,
    threshold: float = 0.5,
    min_size: int = 1,
    weighted: bool = True,
    connectivity: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    単一の異常マップから連結領域ごとの中心座標を抽出する。

    Parameters
    ----------
    anomaly_map : ndarray
        2D の異常スコアマップ。形状 (H, W) または (H, W, 1)。
    threshold : float
        異常スコアの閾値（sigmoid 出力なので 0–1 の範囲）。
    min_size : int
        この画素数より小さい連結領域は除外（ノイズ抑制）。
        2 ピクセル級の異常を確実に拾うなら min_size=2、
        単独の1ピクセル異常も含めるなら min_size=1。
    weighted : bool
        True  → 異常スコアで重み付けした重心（サブピクセル精度。推奨）。
        False → マスクのみの幾何重心。
    connectivity : {4, 8}
        4 連結 or 8 連結。斜めに繋がる異常を 1 つの領域として扱うなら 8。

    Returns
    -------
    centroids : ndarray, shape (N, 2), dtype float32
        各連結領域の中心座標。列は (y, x) の順（画像座標系と整合）。
    sizes : ndarray, shape (N,), dtype int64
        各連結領域の画素数。
    """
    a = np.squeeze(np.asarray(anomaly_map))
    if a.ndim != 2:
        raise ValueError(
            f"2D の異常マップが必要です。実際の形状: {anomaly_map.shape}"
        )

    structure = np.ones((3, 3), bool) if connectivity == 8 else None
    mask = a >= threshold
    labeled, n = ndimage.label(mask, structure=structure)

    if n == 0:
        return (
            np.empty((0, 2), np.float32),
            np.empty((0,), np.int64),
        )

    indices = np.arange(1, n + 1)
    sizes = ndimage.sum(mask, labeled, index=indices).astype(np.int64)

    keep = sizes >= min_size
    if not keep.any():
        return (
            np.empty((0, 2), np.float32),
            np.empty((0,), np.int64),
        )

    indices = indices[keep]
    sizes = sizes[keep]

    # 重み付き / 非重み付き 重心
    src = a if weighted else mask.astype(np.float32)
    centers = np.asarray(
        ndimage.center_of_mass(src, labeled, index=indices),
        dtype=np.float32,
    )
    # 領域が 1 つだけのとき center_of_mass は (y, x) を返すので形を揃える
    if centers.ndim == 1:
        centers = centers[None, :]

    return centers, sizes


def extract_anomaly_centroids_batch(
    anomaly_maps: np.ndarray,
    threshold: float = 0.5,
    min_size: int = 1,
    weighted: bool = True,
    connectivity: int = 8,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    バッチ版。model.predict([target, ref]) の出力 (B, H, W, 1) を直接渡せる。

    Returns
    -------
    list of (centroids, sizes)
        バッチ内の画像ごとに extract_anomaly_centroids の結果のタプル。
    """
    a = np.asarray(anomaly_maps)
    if a.ndim == 4 and a.shape[-1] == 1:
        a = a[..., 0]
    elif a.ndim != 3:
        raise ValueError(
            f"形状 (B, H, W) または (B, H, W, 1) が必要です。"
            f"実際: {anomaly_maps.shape}"
        )

    return [
        extract_anomaly_centroids(
            img, threshold=threshold, min_size=min_size,
            weighted=weighted, connectivity=connectivity,
        )
        for img in a
    ]


# ── 動作確認 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    m = np.zeros((64, 64), dtype=np.float32)

    # ケース1: 単独 1 ピクセル異常
    m[10, 10] = 0.92

    # ケース2: 横並びの 2 ピクセル異常
    m[30, 30] = 0.80
    m[30, 31] = 0.85

    # ケース3: 3x3 の塊（強度に勾配あり → 重み付き重心が中心からずれる）
    m[50:53, 20:23] = rng.uniform(0.6, 0.95, size=(3, 3))

    print("=== weighted = True (推奨) ===")
    centers, sizes = extract_anomaly_centroids(m, threshold=0.5,
                                               min_size=1, weighted=True)
    for (y, x), s in zip(centers, sizes):
        print(f"  center=({y:6.3f}, {x:6.3f})  size={s}")

    print("\n=== weighted = False (マスクのみの幾何重心) ===")
    centers, sizes = extract_anomaly_centroids(m, threshold=0.5,
                                               min_size=1, weighted=False)
    for (y, x), s in zip(centers, sizes):
        print(f"  center=({y:6.3f}, {x:6.3f})  size={s}")
