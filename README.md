# Encofusion
## Encofusion 
Music generation by conditioning with melody and text using <a href="https://github.com/facebookresearch/encodec">Encodec</a> latent expression in VAE.

## Appreciation

- <a href="https://github.com/archinetai/audio-diffusion-pytorch">audio-diffusion-pytorch</a> for The 1D-unet architecture was used.

- <a href="https://github.com/jmoso13/jukebox-diffusion">jukebox-diffusion</a> I was impressed by jukebox-diffusion and refer to its repository. It motivated me to create my first repository.

- <a href="https://github.com/facebookresearch/audiocraft">audiocraft</a> Thank you for the various extensive libraries.
  
## Usage

```install
$ pip install -r requirements.txt
```

```train
$ python train.py --train-data ./dataset/train_data --melody-data ./dataset/melody_data --text-data ./dataset/text_data --batch-size 32 --ckpt-save-location ./ckpts
```

```sampling
$ python sample.py --text <text> --melody <audio path used for melody conditioning> --save_dir
```


