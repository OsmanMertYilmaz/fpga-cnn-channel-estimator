"""Aşama 2: LS kestirimindeki gürültüyü gideren küçük CNN (FP32).

Mimari (ReEsNet çizgisi): 3×3 konvolüsyonlar + artık bağlantı, çıktı = h_ls + düzeltme.
İsteğe bağlı 3. girdi kanalı sabit SNR düzlemidir; artık bağlantı yalnızca gerçek/sanal kanallara eklenir.

Çalıştırma:  .venv/bin/python train.py --name cnn_snr --snr-input --patch 12 --filters 16 16 8
Çıktı:       models/<name>.keras, results/train_<name>.csv
"""
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset import model_inputs

EPOCHS = 60
BATCH = 256
SEED = 0


def build_model(patch_sc=12, n_sym=14, filters=(16, 16, 8), snr_input=False) -> tf.keras.Model:
    inp = tf.keras.Input((patch_sc, n_sym, 3 if snr_input else 2), name="h_ls")
    x = inp
    for i, f in enumerate(filters):
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu", name=f"conv{i + 1}")(x)
    x = tf.keras.layers.Conv2D(2, 3, padding="same", name="conv_out")(x)
    h_ls = inp
    if snr_input:
        # SNR düzlemini artık yoldan ayır: Keras'ta Lambda yerine sabit 1×1 konvolüsyon (hls4ml dostu)
        sel = np.zeros((1, 1, 3, 2), np.float32)
        sel[0, 0, 0, 0] = sel[0, 0, 1, 1] = 1
        pick = tf.keras.layers.Conv2D(2, 1, use_bias=False, trainable=False, name="pick_iq")
        h_ls = pick(inp)
        pick.set_weights([sel])
    out = tf.keras.layers.Add(name="residual")([h_ls, x])
    return tf.keras.Model(inp, out, name="denoise_cnn")


def patch_mse(y_true, y_pred):
    """Yama başına tek MSE değeri; örnek ağırlıkları (N,) şekliyle uygulanabilsin diye."""
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3])


def snr_weight(snr_db):
    """Düz MSE'de düşük SNR örnekleri baskın olur (LS hatası ∝ σ²). Ağırlık ∝ 1/σ² ile kayıp,
    her örneğin LS hatasına göre bağıl hatası gibi davranır ve yüksek SNR da öğrenilir."""
    w = 10 ** (snr_db / 10)
    return w / w.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="cnn_fp32")
    ap.add_argument("--patch", type=int, default=12, help="yama başına alt taşıyıcı (12'nin katı, ≤72)")
    ap.add_argument("--filters", type=int, nargs="+", default=[16, 16, 8])
    ap.add_argument("--snr-input", action="store_true")
    ap.add_argument("--no-weight", action="store_true", help="SNR ağırlıklandırmasını kapat")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    root = Path(__file__).parent
    (root / "models").mkdir(exist_ok=True)
    data = {}
    for split in ("train", "val"):
        d = np.load(root / f"data/{split}.npz")
        group = args.patch // 12
        snr = d["snr"].reshape(-1, group)[:, 0]
        data[split] = (model_inputs(d["x"], d["snr"], args.patch, args.snr_input),
                       d["y"].reshape(-1, args.patch, 14, 2), snr)

    model = build_model(args.patch, 14, tuple(args.filters), args.snr_input)
    print(f"{args.name}: {model.count_params()} parametre, girdi {model.input_shape[1:]}")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=patch_mse)
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(root / f"results/train_{args.name}.csv")),
    ]
    (x, y, snr), (xv, yv, snrv) = data["train"], data["val"]
    w, wv = (None, None) if args.no_weight else (snr_weight(snr), snr_weight(snrv))
    # Büyük yamada örnek sayısı azalır; adım sayısını sabit tutmak için batch'i ölçekle
    batch = max(BATCH * 12 // args.patch, 32)
    model.fit(x, y, sample_weight=w, validation_data=(xv, yv) if wv is None else (xv, yv, wv),
              epochs=EPOCHS, batch_size=batch, callbacks=callbacks, verbose=0)
    model.save(root / f"models/{args.name}.keras")

    pred = model.predict(xv, batch_size=1024, verbose=0)
    ls = xv[..., :2]
    print(f"{args.name} doğrulama: NMSE LS = {10 * np.log10(np.sum((ls - yv) ** 2) / np.sum(yv ** 2)):.2f} dB, "
          f"CNN = {10 * np.log10(np.sum((pred - yv) ** 2) / np.sum(yv ** 2)):.2f} dB")


if __name__ == "__main__":
    main()
