"""Aşama 3: QKeras ile kuantalama farkındalıklı eğitim (QAT).

FP32 modelin ağırlıklarıyla başlar, aynı mimarinin 8 bitlik QKeras karşılığını ince ayarlar.
Girdi, donanımda olduğu gibi ap_fixed<12,4> ızgarasına yuvarlanarak verilir.

Çalıştırma:  .venv/bin/python quantize.py --name cnn_snr_p72
Çıktı:       models/q_<name>.weights.h5, results/qat_<name>.csv
"""
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset import model_inputs
from train import BATCH, patch_mse, snr_weight

EPOCHS = 25
SEED = 0

# Sayı biçimleri (QKeras quantized_bits(bits, integer): işaret biti bits'e dahil, integer'a değil)
IN_BITS, IN_INT = 12, 3            # ap_fixed<12,4>: ±8, adım 1/256
W_BITS = 8                         # ağırlık ve bias: ap_fixed<8,1>: ±1, adım 1/128
ACT = ("quantized_relu(8,2)", "quantized_relu(8,2)", "quantized_relu(8,3)")  # ap_ufixed<8,2>/<8,3>
OUT_BITS, OUT_INT = 10, 2          # düzeltme çıktısı ap_fixed<10,3>: ±4, adım 1/128


def quantize_input(x: np.ndarray) -> np.ndarray:
    """Girdiyi ap_fixed<12,4> ızgarasına yuvarlar ve doyurur (donanıma giden değerler)."""
    step = 2.0 ** -(IN_BITS - 1 - IN_INT)
    lim = 2.0 ** IN_INT
    return np.clip(np.round(x / step) * step, -lim, lim - step).astype(np.float32)


def build_qmodel(fp32: tf.keras.Model) -> tf.keras.Model:
    """FP32 modelin katman adları ve boyutlarıyla aynı QKeras modelini kurar, ağırlıkları kopyalar."""
    # QKeras modül düzeyinde içe aktarılmıyor: QKeras yüklendikten sonra tf.keras.models.load_model
    # (Keras 2.15, .keras dosyası) sonsuz döngüye giriyor. FP32 model her zaman önce yüklenmeli.
    from qkeras import QActivation, QConv2D, quantized_bits
    convs = [l for l in fp32.layers if l.name.startswith("conv") and l.name != "conv_out"]
    wq = quantized_bits(W_BITS, 0, alpha=1)
    inp = tf.keras.Input(fp32.input_shape[1:], name="h_ls")
    x = inp
    for layer, act in zip(convs, ACT):
        x = QConv2D(layer.filters, 3, padding="same", kernel_quantizer=wq, bias_quantizer=wq, name=layer.name)(x)
        x = QActivation(act, name=f"{layer.name}_relu")(x)
    x = QConv2D(2, 3, padding="same", kernel_quantizer=wq, bias_quantizer=wq, name="conv_out")(x)
    x = QActivation(f"quantized_bits({OUT_BITS},{OUT_INT},alpha=1)", name="conv_out_q")(x)
    pick = QConv2D(2, 1, use_bias=False, trainable=False, kernel_quantizer=quantized_bits(2, 1, alpha=1),
                   name="pick_iq")
    out = tf.keras.layers.Add(name="residual")([pick(inp), x])
    q = tf.keras.Model(inp, out, name=f"q_{fp32.name}")
    for layer in fp32.layers:
        if layer.get_weights():
            q.get_layer(layer.name).set_weights(layer.get_weights())
    return q


def load_qmodel(root: Path, name: str) -> tf.keras.Model:
    fp32 = tf.keras.models.load_model(root / f"models/{name}.keras", compile=False)
    q = build_qmodel(fp32)
    q.load_weights(root / f"models/q_{name}.weights.h5")
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="models/<name>.keras FP32 modeli")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    root = Path(__file__).parent
    fp32 = tf.keras.models.load_model(root / f"models/{args.name}.keras", compile=False)
    patch = fp32.input_shape[1]
    q = build_qmodel(fp32)

    data = {}
    for split in ("train", "val"):
        d = np.load(root / f"data/{split}.npz")
        x = model_inputs(d["x"], d["snr"], patch, True)
        x[..., :2] = quantize_input(x[..., :2])
        data[split] = (x, d["y"].reshape(-1, patch, 14, 2), d["snr"].reshape(-1, patch // 12)[:, 0])
    (x, y, snr), (xv, yv, snrv) = data["train"], data["val"]

    def nmse(model):
        p = model.predict(xv, batch_size=256, verbose=0)
        return 10 * np.log10(np.sum((p - yv) ** 2) / np.sum(yv ** 2))

    print(f"{args.name}: FP32 {nmse(fp32):.2f} dB, QAT öncesi 8 bit {nmse(q):.2f} dB")
    q.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss=patch_mse)
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(root / f"results/qat_{args.name}.csv")),
    ]
    q.fit(x, y, sample_weight=snr_weight(snr), validation_data=(xv, yv, snr_weight(snrv)),
          epochs=EPOCHS, batch_size=max(BATCH * 12 // patch, 32), callbacks=callbacks, verbose=0)
    q.save_weights(root / f"models/q_{args.name}.weights.h5")
    print(f"{args.name}: QAT sonrası 8 bit {nmse(q):.2f} dB")


if __name__ == "__main__":
    main()
