import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.filters import (
    to_gray,
    extract_green_channel,
    median_filter,
    histogram_equalization,
    gaussian_blur,
    sobel_filter,
    otsu_threshold,
)
from src.metrics import get_mse, get_psnr, get_ssim


class TestImageProcessingPipeline(unittest.TestCase):
    """Модульні тести для перевірки математичної коректності фільтрів та метрик."""

    def test_color_conversions(self):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb[:, :, 0] = 100
        rgb[:, :, 1] = 150
        rgb[:, :, 2] = 200

        gray = to_gray(rgb)
        green = extract_green_channel(rgb)

        self.assertEqual(gray.shape, (10, 10))
        expected_gray = int(np.clip(0.299 * 100 + 0.587 * 150 + 0.114 * 200, 0, 255))
        self.assertEqual(gray[0, 0], expected_gray)
        self.assertEqual(green[0, 0], 150)

    def test_median_filter_removes_impulse_noise(self):
        img = np.full((7, 7), 120, dtype=np.uint8)
        img[3, 3] = 255

        filtered = median_filter(img, size=3)
        self.assertEqual(filtered[3, 3], 120)

    def test_histogram_equalization_expands_contrast(self):
        low_contrast = np.tile(np.arange(50, 60, dtype=np.uint8), (10, 1))
        equalized = histogram_equalization(low_contrast)

        self.assertEqual(equalized.shape, low_contrast.shape)
        self.assertEqual(int(equalized.min()), 0)
        self.assertEqual(int(equalized.max()), 255)

    def test_gaussian_and_sobel_edge_detection(self):
        step_img = np.zeros((20, 20), dtype=np.uint8)
        step_img[:, 10:] = 255

        blurred = gaussian_blur(step_img, size=3, sigma=1.0)
        edges = sobel_filter(blurred)

        self.assertGreater(int(edges[:, 9:11].max()), 200)
        self.assertEqual(int(edges[:, 0].max()), 0)

    def test_otsu_threshold_bimodal(self):
        bimodal = np.zeros((10, 10), dtype=np.uint8)
        bimodal[:, :5] = 40
        bimodal[:, 5:] = 210

        binary = otsu_threshold(bimodal)
        self.assertTrue(np.all(binary[:, :5] == 0))
        self.assertTrue(np.all(binary[:, 5:] == 255))

    def test_quality_metrics(self):
        img = np.full((16, 16), 100, dtype=np.uint8)
        shifted = np.full((16, 16), 110, dtype=np.uint8)

        self.assertEqual(get_mse(img, img), 0.0)
        self.assertEqual(get_psnr(img, img), float("inf"))
        self.assertAlmostEqual(get_ssim(img, img), 1.0, places=6)

        self.assertAlmostEqual(get_mse(img, shifted), 100.0, places=6)
        self.assertLess(get_ssim(img, shifted), 1.0)


if __name__ == "__main__":
    unittest.main()