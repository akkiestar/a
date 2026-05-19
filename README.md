# FractalDB-10k ResNet-50 + ArcFace (TensorFlow)

法務的に完全クリーンな FractalDB-10k 事前学習済み ResNet-50 を起点に、ArcFace で
メトリック学習を行い、プロトタイプ + cos 類似度で 25 クラス分類するパイプライン。

## 前提

- TensorFlow 2.10+（GPU 推奨）
- PyTorch（重み変換時のみ）

```bash
pip install -r requirements.txt
```

## 1. FractalDB 事前学習済み重みのダウンロード

[FractalDB-Pretrained-ResNet-PyTorch](https://github.com/hirokatsukataoka16/FractalDB-Pretrained-ResNet-PyTorch)
のリポジトリから `FractalDB-10000_resnet50_epoch90.pth` 等をダウンロード。

## 2. PyTorch → TensorFlow 重み変換

```bash
python convert_fractaldb_to_tf.py \
    --pt_weights FractalDB-10000_resnet50_epoch90.pth \
    --tf_output fractaldb10k_resnet50.weights.h5
```

## 3. CSV の準備

`train.csv`, `val.csv`, `test.csv` を以下の形式で用意:

```csv
image_path,label
/path/to/img_001.jpg,0
/path/to/img_002.jpg,3
/path/to/img_003.jpg,12
```

- `image_path`: 画像への絶対パスまたは相対パス
- `label`: 0 〜 (num_classes-1) の整数 ID

## 4. 学習

```bash
python train.py \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --pretrained_weights fractaldb10k_resnet50.weights.h5 \
    --num_classes 25 \
    --output_dir ./output \
    --epochs 50 \
    --batch_size 64 \
    --image_size 224 \
    --lp_epochs 3 \
    --base_lr 0.01
```

学習中、各エポックで以下を計算:
- 訓練データ全体でプロトタイプ（クラスごとの埋め込み平均）を計算
- 検証データの各画像を最近傍プロトタイプに分類して accuracy を算出
- ベスト時点の埋め込みモデルを `output/best_embedding.weights.h5` に保存

学習完了後、`output/prototypes.npy` にプロトタイプ（shape: `[num_classes, embedding_dim]`）を保存。

## 5. 評価

```bash
python evaluate.py \
    --test_csv data/test.csv \
    --weights ./output/best_embedding.weights.h5 \
    --prototypes ./output/prototypes.npy \
    --num_classes 25 \
    --unknown_threshold 0.0
```

- `--unknown_threshold` を 0 より大きい値（例: 0.3）にすると、最近傍類似度が
  しきい値未満のサンプルを「未分類」として扱う（オープンセット運用）

## チューニングのポイント

- **データ不均衡**: 各クラスのサンプル数に大きな偏りがある場合、`pd.DataFrame.sample(weights=...)` で
  バランスサンプリングするか、focal loss に切り替える
- **入力サイズ**: 224px で精度不足なら 288 / 320 px を試す。Jetson での速度予算と相談
- **ArcFace パラメータ**: 学習が不安定なら `--arcface_margin 0.3`、収束が遅ければ `--arcface_scale 64`
- **augmentation**: `data.py` の `_augment` を強める / 弱める

## Jetson 展開

学習後の `embedding_model` を SavedModel → ONNX → TensorRT (INT8) に変換して使用。
TensorRT 量子化後はプロトタイプを再計算するのが安全。

```python
embedding_model.save("./output/embedding_savedmodel")
# python -m tf2onnx.convert --saved-model ./output/embedding_savedmodel --output embedding.onnx --opset 13
# trtexec --onnx=embedding.onnx --saveEngine=embedding_int8.engine --int8 --calib=...
```

## ファイル構成

| ファイル | 役割 |
|---|---|
| `convert_fractaldb_to_tf.py` | PyTorch 重みを TF 形式に変換 |
| `model.py` | ResNet-50 + ArcFace モデル定義 |
| `data.py` | CSV → tf.data パイプライン |
| `train.py` | 学習スクリプト（LP-FT + cosine LR + k-NN 検証） |
| `evaluate.py` | テスト評価と混同行列出力 |
