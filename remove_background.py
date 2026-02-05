import cv2
import numpy as np
from rembg import remove
from rembg.bg import new_session
from PIL import Image
import io
from typing import Optional

class SharpestCutout:
    def __init__(self, image_path: str, model_name: str = "u2net", alpha_matting: bool = False):
        self.image_path = image_path
        self.model_name = model_name
        self.alpha_matting = alpha_matting
        self.session = new_session(self.model_name)
        self.fg: Optional[np.ndarray] = None
        self._load_and_remove_background()

    def _load_image_bytes(self) -> bytes:
        with open(self.image_path, "rb") as f:
            return f.read()

    def _remove_background(self, image_bytes: bytes) -> bytes:
        return remove(image_bytes, session=self.session, alpha_matting=self.alpha_matting)

    def _convert_to_numpy(self, image_bytes: bytes) -> np.ndarray:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        np_image = np.array(pil_image)
        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2BGRA)
        return np_image

    def _load_and_remove_background(self) -> None:
        image_bytes = self._load_image_bytes()
        output_bytes = self._remove_background(image_bytes)
        self.fg = self._convert_to_numpy(output_bytes)

    def show(self, window_name: str = "Sharpest Cutout") -> None:
        if self.fg is not None:
            cv2.imshow(window_name, self.fg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def save(self, save_path: str) -> None:
        if self.fg is not None:
            cv2.imwrite(save_path, self.fg)

if __name__ == "__main__":
    IMAGE_PATH = "ChatGPT Image Jul 29, 2025 at 11_14_15 PM.png"
    OUTPUT_PATH = "cutout_output.png"
    remover = SharpestCutout(IMAGE_PATH)
    remover.show()
    remover.save(OUTPUT_PATH)
