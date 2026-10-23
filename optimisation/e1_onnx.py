"""
Expérience 1 — Export ONNX + PyTorch vs ONNX Runtime (CPU)

Question : la conversion en ONNX change-t-elle le modèle ? Le moteur ONNX est-il plus rapide ?
Rôle : crée la RÉFÉRENCE non quantifiée (ONNX FP32) qui isole ensuite l'effet de la quantization.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from optimisation.common import (CFG, P, banner, batch_of, bench, evaluate, export_onnx,  # noqa: E402
                                 images_of, save_json, session, size_mb, test_split)

IMGSZ = 640


def main():
    onnx_path = export_onnx(IMGSZ)
    net = YOLO(str(P["model_pt"])).model.float().eval().cpu()
    sess = session(onnx_path)
    name = sess.get_inputs()[0].name

    def run_torch(x):
        with torch.inference_mode():
            return net(torch.from_numpy(x))[0].numpy()

    def run_onnx(x):
        return sess.run(None, {name: x})[0]

    split = test_split()
    imgs = batch_of(images_of(split)[:16], IMGSZ)
    d = abs(run_torch(imgs)[:, 4:, :] - run_onnx(imgs)[:, 4:, :])

    map_t = evaluate(P["model_pt"], IMGSZ, tag="pytorch", project="e1")
    map_o = evaluate(onnx_path, IMGSZ, tag="onnx_fp32", project="e1")

    speed = {"pytorch": {}, "onnx": {}}
    for bs in CFG["benchmark"]["batches"]:
        speed["pytorch"][bs] = bench(run_torch, IMGSZ, bs)
        speed["onnx"][bs] = bench(run_onnx, IMGSZ, bs)

    res = {"split": split,
           "pytorch": {"taille_mo": size_mb(P["model_pt"]), "mAP50": map_t[0], "mAP50-95": map_t[1],
                       "ms_par_image": speed["pytorch"]},
           "onnx_fp32": {"taille_mo": size_mb(onnx_path), "mAP50": map_o[0], "mAP50-95": map_o[1],
                         "ms_par_image": speed["onnx"]},
           "ecart_sorties": {"max": float(d.max()), "moyen": float(d.mean())}}
    save_json("etape1.json", res)

    p, o = res["pytorch"], res["onnx_fp32"]
    lines = banner("E1 — PyTorch vs ONNX Runtime (CPU)") + [
        f"{'':18}{'PyTorch':>12}{'ONNX FP32':>12}",
        f"{'Taille (Mo)':18}{p['taille_mo']:>12.2f}{o['taille_mo']:>12.2f}",
        f"{'mAP50':18}{p['mAP50']:>12.3f}{o['mAP50']:>12.3f}",
        f"{'mAP50-95':18}{p['mAP50-95']:>12.3f}{o['mAP50-95']:>12.3f}"]
    for bs in CFG["benchmark"]["batches"]:
        lines.append(f"{'ms/img batch=' + str(bs):18}{p['ms_par_image'][bs]:>12.1f}{o['ms_par_image'][bs]:>12.1f}")
    lines.append(f"Écart des scores : max {d.max():.2e} | moyen {d.mean():.2e}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
