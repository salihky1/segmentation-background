import cv2
import numpy as np
from typing import List, Tuple

def read_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at '{image_path}'")
    return image

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_threshold(gray_image: np.ndarray, threshold_value: int = 200, max_value: int = 300) -> np.ndarray:
    _, thresh = cv2.threshold(gray_image, threshold_value, max_value, cv2.THRESH_BINARY)
    return thresh

def find_contours(thresh_image: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def compute_convex_hulls(contours: List[np.ndarray]) -> List[np.ndarray]:
    return [cv2.convexHull(cnt) for cnt in contours]

def draw_contours_and_hulls(image_shape: Tuple[int, int, int], contours: List[np.ndarray], hulls: List[np.ndarray]) -> np.ndarray:
    result = np.zeros(image_shape, np.uint8)
    for i in range(len(contours)):
        cv2.drawContours(result, contours, i, (255, 0, 0), 3)
        cv2.drawContours(result, hulls, i, (0, 255, 0), 1)
    return result

def show_image(window_name: str, image: np.ndarray) -> None:
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image(image_path: str) -> None:
    image = read_image(image_path)
    gray = convert_to_grayscale(image)
    thresh = apply_threshold(gray)
    contours, _ = find_contours(thresh)
    hulls = compute_convex_hulls(contours)
    result = draw_contours_and_hulls(image.shape, contours, hulls)
    show_image("Contours and Convex Hulls", result)

if __name__ == "__main__":
    IMAGE_PATH = "yildiz.jpg"
    process_image(IMAGE_PATH)
