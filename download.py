import os
import shutil

import gdown


URL = "https://drive.google.com/uc?id=12_1NbybiYigqL4pJP8r9ZnaY621dpCaL"


def download():
    gdown.download(URL)

    os.makedirs("src/weights", exist_ok=True)
    shutil.move("hifi.pth", "src/weights/hifi.pth")


if __name__ == "__main__":
    download()