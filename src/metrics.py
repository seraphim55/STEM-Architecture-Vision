import numpy as np


def _validate_inputs(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Перевіряє сумісність розмірів матриць та переводить їх у float64."""
    if img1.shape != img2.shape:
        raise ValueError(f"Розміри зображень не збігаються: {img1.shape} != {img2.shape}")
    return img1.astype(np.float64), img2.astype(np.float64)


def get_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Обчислює середньоквадратичну похибку (MSE) між двома зображеннями."""
    a, b = _validate_inputs(img1, img2)
    return float(np.mean((a - b) ** 2))


def get_psnr(img1: np.ndarray, img2: np.ndarray, max_pixel: float = 255.0) -> float:
    """Обчислює пікове відношення сигнал/шум (PSNR) у децибелах."""
    mse = get_mse(img1, img2)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((max_pixel ** 2) / mse))


def get_ssim(img1: np.ndarray, img2: np.ndarray, max_pixel: float = 255.0) -> float:
    """Обчислює індекс структурної подібності (SSIM)."""
    a, b = _validate_inputs(img1, img2)

    c1 = (0.01 * max_pixel) ** 2
    c2 = (0.03 * max_pixel) ** 2

    mu_x = np.mean(a)
    mu_y = np.mean(b)

    sigma_x2 = np.var(a)
    sigma_y2 = np.var(b)
    sigma_xy = np.mean((a - mu_x) * (b - mu_y))

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x2 + sigma_y2 + c2)

    return float(numerator / denominator)