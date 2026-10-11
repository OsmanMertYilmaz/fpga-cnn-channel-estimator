"""Aşama 4: QKeras modelini hls4ml ile HLS C++ projesine çevirir ve bit-bit doğrular.

Mac'te: dönüştürme + C++ derleme + hls_model.predict() ile QKeras çıktısını bit-bit karşılaştırma.
Linux'ta (Vitis HLS kurulu): --build ile C-sim, sentez ve C/RTL co-simulation.

Çalıştırma:  .venv/bin/python hls_convert.py --name cnn_snr_p72 [--build] [--reuse 32]
Çıktı:       hls/<name>/ (HLS projesi, tb_data/ test vektörleri dahil)
"""
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset import model_inputs
from quantize import IN_BITS, IN_INT, OUT_BITS, OUT_INT, load_qmodel, quantize_input

PART = "xczu7ev-ffvc1156-2-e"      # ZCU104
CLOCK_NS = 5.0                     # 200 MHz
N_TB_GRIDS = 20                    # C-sim/co-sim test vektörü sayısı (co-sim yavaş)
N_CHECK_GRIDS = 500                # Mac'te bit-bit karşılaştırma için ızgara sayısı

# Toplayıcılar kayıpsız: 8 bit ağırlık (6..7 kesir) × ≤12 bit girdi (8 kesir) → ≤15 kesir biti yeter,
# 3×3×32 terimlik toplam için ±2^13 aralık bol. Kuantalama yalnızca QActivation katmanlarında olur.
ACCUM = "fixed<32,14>"


def hls_config(hls4ml, qmodel, reuse: int) -> dict:
    cfg = hls4ml.utils.config_from_keras_model(qmodel, granularity="name", backend="Vitis")
    cfg["Model"]["Strategy"] = "Resource"
    cfg["Model"]["ReuseFactor"] = reuse
    layers = cfg["LayerName"]
    layers["h_ls"]["Precision"]["result"] = f"fixed<{IN_BITS},{IN_INT + 1},RND_CONV,SAT>"
    for name, lc in layers.items():
        if name.startswith("conv") and not name.endswith(("_relu", "_q")) or name == "pick_iq":
            lc["Precision"]["result"] = ACCUM
            lc["Precision"]["accum"] = ACCUM
    layers["residual"]["Precision"]["result"] = f"fixed<{max(IN_BITS, OUT_BITS) + 1},{max(IN_INT, OUT_INT) + 2}>"
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--reuse", type=int, default=32, help="ReuseFactor (yalnızca sentezi etkiler)")
    ap.add_argument("--build", action="store_true", help="Vitis HLS ile csim + sentez + co-sim (Linux)")
    args = ap.parse_args()

    root = Path(__file__).parent
    qmodel = load_qmodel(root, args.name)   # QKeras'tan önce FP32 yüklenir (bkz. quantize.py)
    import hls4ml                           # hls4ml QKeras'ı içe aktarır; bu yüzden burada

    patch = qmodel.input_shape[1]
    d = np.load(root / "data/val.npz")
    x = quantize_input(model_inputs(d["x"], d["snr"], patch, True))   # SNR düzlemi dahil 3 kanal
    y_true = d["y"].reshape(-1, patch, 14, 2)
    n = N_CHECK_GRIDS * 72 // patch
    x, y_true = x[:n], y_true[:n]
    y_q = qmodel.predict(x, batch_size=256, verbose=0)

    out_dir = root / "hls" / args.name
    tb_dir = root / "hls" / f"{args.name}_tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    n_tb = N_TB_GRIDS * 72 // patch
    np.save(tb_dir / "x.npy", x[:n_tb])
    np.save(tb_dir / "y.npy", y_q[:n_tb])

    hm = hls4ml.converters.convert_from_keras_model(
        qmodel, hls_config=hls_config(hls4ml, qmodel, args.reuse), backend="Vitis", io_type="io_stream",
        output_dir=str(out_dir), project_name="chest", part=PART, clock_period=CLOCK_NS,
        input_data_tb=str(tb_dir / "x.npy"), output_data_tb=str(tb_dir / "y.npy"))
    hm.compile()
    y_h = hm.predict(np.ascontiguousarray(x)).reshape(y_q.shape)

    diff = np.abs(y_h - y_q)
    nmse = lambda p: 10 * np.log10(np.sum((p - y_true) ** 2) / np.sum(y_true ** 2))
    print(f"{args.name}: {diff.size} çıktı değeri, bit-bit eşleşme %{100 * np.mean(diff == 0):.3f}, "
          f"max fark {diff.max():.2e}")
    print(f"{args.name}: NMSE QKeras {nmse(y_q):.3f} dB, hls4ml C++ {nmse(y_h):.3f} dB")

    if args.build:
        report = hm.build(csim=True, synth=True, cosim=True, validation=True)
        hls4ml.report.read_vivado_report(str(out_dir))
        print(report)


if __name__ == "__main__":
    main()
