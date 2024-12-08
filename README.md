# HIFI-GAN

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains the custom realization of HIFI-GAN vocoder

## Installation

Follow these steps to run the project:

## How To Use

0. Clone repository

```bash
https://github.com/AntonNuzhdin/hifi-gan_vocoder
cd hifi-gan_vocoder
```
1. Create and activate env

```bash
conda create -n hifi python=3.11.10

conda activate hifi
```

2. Install requirements

```bash
pip install -r requirements.txt
```

3. Dowload models weights

```bash
python download.py 
```

4. Run synthesize

To run the HIFI-GAN you can run the scrtipt providing end-to-end speech synthesize using Tacotron + HIFI-GAN (ours)
```bash
python synthesize.py text="zvuk zvuuuk zvuuuuuuuk"
```

To custimize the synthesizing, feel free to edit the src/configs/custom_dir.yaml

```
model:
  hifigan_checkpoint: src/weights/convtasnet.pth
  output_dir: ./predictions # Where to store the predictions

data:
  data_dir: None # The directory with .txt files
  extension: .txt

text: "Dmitri Shostakovich was a Soviet-era Russian composer and pianist who became internationally known after the premiere of his First Symphony in 1926 and thereafter was regarded as a major composer." # Text to generate
```

After running synthesize.py the WV-MOS metric of generated audio will be printed. 

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)


## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
