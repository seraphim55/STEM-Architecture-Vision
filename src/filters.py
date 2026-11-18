import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    """Перетворює RGB-зображення у напівтонове за стандартом ITU-R BT.601."""
    if img.ndim == 2:
        return img.astype(np.uint8)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


def extract_green_channel(img: np.ndarray) -> np.ndarray:
    """Виділяє зелений спектральний канал із RGB-зображення."""
    if img.ndim == 2:
        return img.astype(np.uint8)
    return img[:, :, 1].astype(np.uint8)


def pad_image(img: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """Доповнює межі матриці дублюванням крайніх пікселів."""
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")


def median_filter(img: np.ndarray, size: int = 3) -> np.ndarray:
    """Виконує медіанну фільтрацію для приглушення імпульсного шуму."""
    if size % 2 == 0:
        raise ValueError("Розмір вікна медіанного фільтра має бути непарним.")
    pad = size // 2
    padded = pad_image(img, pad, pad)
    h, w = img.shape
    windows = [
        padded[i:i + h, j:j + w]
        for i in range(size)
        for j in range(size)
    ]
    return np.median(np.stack(windows, axis=0), axis=0).astype(np.uint8)


def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """Вирівнює контраст зображення через кумулятивну функцію розподілу (CDF)."""
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    cdf = np.cumsum(hist)
    non_zero = cdf[cdf > 0]
    if non_zero.size == 0:
        return img.copy()

    cdf_min = non_zero[0]
    total_pixels = img.size
    if total_pixels == cdf_min:
        return img.copy()

    lut = np.round(((cdf - cdf_min) / (total_pixels - cdf_min)) * 255.0)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[img]