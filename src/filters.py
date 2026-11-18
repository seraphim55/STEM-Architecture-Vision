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


def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Виконує дискретну 2D-згортку матриці зображення з ядром фільтра."""
    kernel = np.flipud(np.fliplr(kernel))
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = pad_image(img, pad_h, pad_w).astype(np.float64)
    h, w = img.shape

    output = np.zeros((h, w), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            output += kernel[i, j] * padded[i:i + h, j:j + w]
    return output


def gaussian_blur(img: np.ndarray, size: int = 5, sigma: float = 1.4) -> np.ndarray:
    """Згладжує високочастотні шуми за допомогою двовимірного ядра Гауса."""
    if size % 2 == 0:
        raise ValueError("Розмір ядра Гауса має бути непарним.")
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)

    blurred = convolve2d(img, kernel)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def sobel_filter(img: np.ndarray) -> np.ndarray:
    """Обчислює модуль градієнта яскравості оператором Собеля для виділення контурів."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    gx = convolve2d(img, kx)
    gy = convolve2d(img, ky)
    magnitude = np.hypot(gx, gy)

    max_val = np.max(magnitude)
    if max_val > 0:
        magnitude = (magnitude / max_val) * 255.0
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def otsu_threshold(img: np.ndarray) -> np.ndarray:
    """Виконує автоматичну порогову бінаризацію методом Оцу."""
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    total = img.size
    current_max = -1.0
    threshold = 0

    sum_total = np.dot(np.arange(256, dtype=np.float64), hist)
    sum_b = 0.0
    w_b = 0

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break

        sum_b += t * hist[t]
        mean_b = sum_b / w_b
        mean_f = (sum_total - sum_b) / w_f

        between_var = w_b * w_f * (mean_b - mean_f) ** 2
        if between_var > current_max:
            current_max = between_var
            threshold = t

    return np.where(img > threshold, 255, 0).astype(np.uint8)