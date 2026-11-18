import csv
import json
import os
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.filters import (
    extract_green_channel,
    gaussian_blur,
    histogram_equalization,
    median_filter,
    otsu_threshold,
    sobel_filter,
    to_gray,
)
from src.metrics import get_mse, get_psnr, get_ssim


def load_config(path: str = "config.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_comparison_strip(
    gray: np.ndarray, p1: np.ndarray, p2: np.ndarray, save_path: str
) -> None:
    """Зберігає склейку 1:1 (Оригінал | Пайплайн 1 | Пайплайн 2) для звіту."""
    divider = np.full((gray.shape[0], 4), 255, dtype=np.uint8)
    combined = np.hstack([gray, divider, p1, divider, p2])
    Image.fromarray(combined, mode="L").save(save_path)


def generate_telemetry_plots(records: list[dict], plot_path: str) -> None:
    """Будує графіки метрик якості та часу виконання для всього датасету."""
    indices = np.arange(1, len(records) + 1)

    p1_psnr = [r["p1_psnr"] for r in records]
    p2_psnr = [r["p2_psnr"] for r in records]
    p1_ssim = [r["p1_ssim"] for r in records]
    p2_ssim = [r["p2_ssim"] for r in records]
    p1_mse = [r["p1_mse"] for r in records]
    p2_mse = [r["p2_mse"] for r in records]
    p1_time = [r["p1_time_ms"] for r in records]
    p2_time = [r["p2_time_ms"] for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), dpi=150)
    fig.suptitle(
        "Аналіз ефективності конвеєрів обробки зображень (N = 120)",
        fontsize=13,
        fontweight="bold",
    )

    # 1. PSNR
    ax = axes[0, 0]
    ax.plot(indices, p1_psnr, label="Пайплайн 1 (Median + HistEq)", color="#1f77b4", lw=1.3)
    ax.plot(indices, p2_psnr, label="Пайплайн 2 (Gauss + Sobel + Otsu)", color="#d62728", lw=1.3)
    ax.axhline(np.mean(p1_psnr), color="#1f77b4", ls="--", alpha=0.6)
    ax.axhline(np.mean(p2_psnr), color="#d62728", ls="--", alpha=0.6)
    ax.set_title("Пікове відношення сигнал/шум (PSNR, дБ)")
    ax.set_xlabel("Індекс зображення")
    ax.set_ylabel("PSNR (дБ)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 2. SSIM
    ax = axes[0, 1]
    ax.plot(indices, p1_ssim, label="Пайплайн 1 (Median + HistEq)", color="#1f77b4", lw=1.3)
    ax.plot(indices, p2_ssim, label="Пайплайн 2 (Gauss + Sobel + Otsu)", color="#d62728", lw=1.3)
    ax.axhline(np.mean(p1_ssim), color="#1f77b4", ls="--", alpha=0.6)
    ax.axhline(np.mean(p2_ssim), color="#d62728", ls="--", alpha=0.6)
    ax.set_title("Індекс структурної подібності (SSIM)")
    ax.set_xlabel("Індекс зображення")
    ax.set_ylabel("SSIM")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 3. MSE
    ax = axes[1, 0]
    ax.plot(indices, p1_mse, label="Пайплайн 1", color="#1f77b4", lw=1.3)
    ax.plot(indices, p2_mse, label="Пайплайн 2", color="#d62728", lw=1.3)
    ax.set_title("Середньоквадратична похибка (MSE)")
    ax.set_xlabel("Індекс зображення")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 4. Час виконання
    ax = axes[1, 1]
    ax.plot(indices, p1_time, label="Пайплайн 1", color="#2ca02c", lw=1.2)
    ax.plot(indices, p2_time, label="Пайплайн 2", color="#ff7f0e", lw=1.2)
    ax.set_title("Час обробки одного кадру (мс)")
    ax.set_xlabel("Індекс зображення")
    ax.set_ylabel("Час (мс)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(plot_path)
    plt.close(fig)


def main():
    cfg = load_config()
    dataset_dir = cfg["dataset_dir"]
    output_dir = cfg["output_dir"]
    samples_dir = os.path.join(output_dir, "samples")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    median_size = cfg["median_size"]
    gauss_size = cfg["gaussian"]["size"]
    gauss_sigma = cfg["gaussian"]["sigma"]

    files = sorted(
        f for f in os.listdir(dataset_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not files:
        raise FileNotFoundError(f"У папці '{dataset_dir}' не знайдено зображень.")

    records = []
    for idx, filename in enumerate(files, start=1):
        img_path = os.path.join(dataset_dir, filename)
        with Image.open(img_path) as img:
            rgb = np.array(img.convert("RGB"), dtype=np.uint8)

        ref_gray = to_gray(rgb)

        # Пайплайн 1: Grayscale -> Median Filter -> Histogram Equalization
        t0 = time.perf_counter()
        gray_p1 = to_gray(rgb)
        denoised = median_filter(gray_p1, size=median_size)
        p1_res = histogram_equalization(denoised)
        p1_ms = (time.perf_counter() - t0) * 1000.0

        # Пайплайн 2: Green Channel -> Gaussian Blur -> Sobel -> Otsu
        t0 = time.perf_counter()
        green_p2 = extract_green_channel(rgb)
        blurred = gaussian_blur(green_p2, size=gauss_size, sigma=gauss_sigma)
        edges = sobel_filter(blurred)
        p2_res = otsu_threshold(edges)
        p2_ms = (time.perf_counter() - t0) * 1000.0

        records.append(
            {
                "filename": filename,
                "p1_mse": round(get_mse(ref_gray, p1_res), 4),
                "p1_psnr": round(get_psnr(ref_gray, p1_res), 4),
                "p1_ssim": round(get_ssim(ref_gray, p1_res), 4),
                "p1_time_ms": round(p1_ms, 3),
                "p2_mse": round(get_mse(ref_gray, p2_res), 4),
                "p2_psnr": round(get_psnr(ref_gray, p2_res), 4),
                "p2_ssim": round(get_ssim(ref_gray, p2_res), 4),
                "p2_time_ms": round(p2_ms, 3),
            }
        )

        if idx <= 5:
            sample_name = f"comparison_{idx:03d}.png"
            save_comparison_strip(
                ref_gray, p1_res, p2_res, os.path.join(samples_dir, sample_name)
            )

    # Експорт у CSV
    csv_path = cfg["csv_report"]
    fieldnames = list(records[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Побудова графіків
    plot_path = cfg["plot_report"]
    generate_telemetry_plots(records, plot_path)

    print(f"Оброблено зображень: {len(records)}")
    print(
        f"Пайплайн 1 (середні): MSE={np.mean([r['p1_mse'] for r in records]):.2f}, "
        f"PSNR={np.mean([r['p1_psnr'] for r in records]):.2f} дБ, "
        f"SSIM={np.mean([r['p1_ssim'] for r in records]):.4f}, "
        f"Час={np.mean([r['p1_time_ms'] for r in records]):.2f} мс"
    )
    print(
        f"Пайплайн 2 (середні): MSE={np.mean([r['p2_mse'] for r in records]):.2f}, "
        f"PSNR={np.mean([r['p2_psnr'] for r in records]):.2f} дБ, "
        f"SSIM={np.mean([r['p2_ssim'] for r in records]):.4f}, "
        f"Час={np.mean([r['p2_time_ms'] for r in records]):.2f} мс"
    )


if __name__ == "__main__":
    main()  