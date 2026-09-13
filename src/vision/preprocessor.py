from __future__ import annotations

import cv2
import numpy as np


class NameplatePreprocessor:
    def __init__(self, blur_threshold: float = 100.0):
        self.blur_threshold = blur_threshold

    def check_blur(self, gray_image: np.ndarray) -> tuple[bool, float]:
        """
        Computes the variance of the Laplacian to quantify focus sharpness.
        Lower values indicate significant camera or motion blur.
        """
        laplacian_var = float(cv2.Laplacian(gray_image, cv2.CV_64F).var())
        is_clear = laplacian_var >= self.blur_threshold
        return is_clear, laplacian_var

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Sorts 4 coordinates in consistent topological order:
        [top-left, top-right, bottom-right, bottom-left].
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum (x + y)
        rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right has smallest diff (y - x)
        rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest diff

        return rect

    def find_nameplate_corners(self, gray_image: np.ndarray) -> np.ndarray | None:
        """
        Detects prominent rectangular boundaries on industrial equipment.
        Uses bilateral filtering to suppress metallic texture noise while preserving edges.
        """
        filtered = cv2.bilateralFilter(gray_image, d=9, sigmaColor=75, sigmaSpace=75)
        edges = cv2.Canny(filtered, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        image_area = gray_image.shape[0] * gray_image.shape[1]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter: the nameplate should occupy between 5% and 95% of the total frame
            if 0.05 * image_area < area < 0.95 * image_area:
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

                # A valid nameplate candidate is convex and has exactly 4 vertices
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    return approx.reshape(4, 2).astype("float32")

        return None

    def warp_perspective(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Applies a 4-point homography transform to flatten perspective tilt.
        """
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        # Calculate maximum width and height using Euclidean distances
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(int(height_a), int(height_b))

        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )

        matrix = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, matrix, (max_width, max_height))

    def enhance_for_ocr(self, warped_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Upscales small text patches and applies edge-preserving denoising
        with gentle contrast normalization.
        """
        # 1. Bicubic 2x upscaling so CRAFT sees 30-40px character heights
        h, w = warped_bgr.shape[:2]
        upscaled = cv2.resize(warped_bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

        # 2. Edge-preserving bilateral filter (suppresses texture noise without blurring text edges)
        denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

        # 3. Controlled CLAHE (clipLimit lowered from 3.0 to 1.8 to prevent noise amplification)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        contrast_adjusted = clahe.apply(denoised)

        # 4. Otsu adaptive binarization
        _, binarized = cv2.threshold(contrast_adjusted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return contrast_adjusted, binarized

    def process(self, image_bgr: np.ndarray) -> dict:
        """
        Executes the vision pipeline and returns intermediate diagnostic images.
        """
        gray_raw = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        is_clear, blur_score = self.check_blur(gray_raw)

        corners = self.find_nameplate_corners(gray_raw)

        if corners is not None:
            warped = self.warp_perspective(image_bgr, corners)
            rectified = True
        else:
            warped = image_bgr.copy()
            rectified = False

        enhanced_gray, binarized = self.enhance_for_ocr(warped)

        return {
            "blur_passed": is_clear,
            "blur_score": blur_score,
            "corners_detected": rectified,
            "warped_color": warped,
            "enhanced_gray": enhanced_gray,
            "binarized": binarized,
        }