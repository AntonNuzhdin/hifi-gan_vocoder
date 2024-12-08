import os
import shutil

import gdown

URL = "https://drive.google.com/uc?id=1_2sbOLFPOLOl6rPHa5UxWYpq0OWphY0H"


def download():
    gdown.download(URL)

    os.makedirs("src/weights", exist_ok=True)
    shutil.move("hifi.pth", "src/weights/hifi.pth")


if __name__ == "__main__":
    download()