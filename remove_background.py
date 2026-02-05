import cv2
import numpy as np
from rembg import remove
from rembg.bg import new_session
from PIL import Image
import io

class SharpestCutout:
    def __init__(self, image_path):
        self.image_path = image_path
        self.session = new_session("u2net")  # en güçlü model

        with open(self.image_path, "rb") as f:
            input_image = f.read()

        # EN ÖNEMLİ AYAR BURADA:
        # alpha_matting=False => blur yok, kenarlar sert
        output = remove(
            input_image,
            session=self.session,
            alpha_matting=False
        )

        # Görseli RGBA olarak aç
        pil_image = Image.open(io.BytesIO(output)).convert("RGBA")
        self.fg = np.array(pil_image)
        self.fg = cv2.cvtColor(self.fg, cv2.COLOR_RGBA2BGRA)

    def show(self):
        cv2.imshow("Sert Kenarlı Arka Plan Kaldırma", self.fg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Kullanım:
remover = SharpestCutout("ChatGPT Image Jul 29, 2025 at 11_14_15 PM.png")
remover.show()
