import os 
import torch
import argparse
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from audiocraft.models import builders, MusicGen
from utils.audio_diffusion import AudioDiffusion
from utils.utils import (read_yaml_file, parse_diff_conf, my_collate, ExceptionCallback, AudioDataset)

device = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG_FILE = "./configs/audio_diffusion/diffusion.yaml"

def run(*args, **kwargs):
    conf = read_yaml_file(CONFIG_FILE)
    audio_dir = kwargs["train_data"]
    melody_dir = kwargs["melody_data"]
    text_dir = kwargs["text_data"]
    save_path = kwargs["ckpt_save_location"]
    log_to_wandb = kwargs["log_to_wandb"]
    resume_ckpt = kwargs['resume_ckpt']
    num_workers = kwargs['num_workers']
    batch_size = kwargs['batch_size']
    ckpt_every = kwargs['ckpt_every']
    proj_name = kwargs['project_name']
    use_cfg = kwargs['use_cfg']
    
    music_gen = MusicGen.get_pretrained()
    compression_model = music_gen.compression_model
    lm = music_gen.lm
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=ckpt_every, save_top_k=-1, dirpath=save_path)
    
    diffusion_conf = conf["model"]["diffusion"]
    diffusion_conf = parse_diff_conf(diffusion_conf)
    
    dataset = AudioDataset(audio_data=audio_dir, text_data=text_dir, melody_data=melody_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, 
                            pin_memory=False, drop_last=True, collate_fn=my_collate)
    diffusion_model = AudioDiffusion(compression_model, lm, diffusion_kwargs=diffusion_conf,
                                     use_cfg=use_cfg)
    
    if log_to_wandb:
        wandb_logger = pl.loggers.WandbLogger(project=proj_name, log_model='all')
        wandb_logger.watch(diffusion_model)
        diffusion_trainer = pl.Trainer(
            devices=1,
            accelerator="gpu",
            precision=16,
            accumulate_grad_batches=4, 
            callbacks=[ckpt_callback, exc_callback],
            logger=wandb_logger,
            log_every_n_steps=1,
            max_epochs=10000000,
            )
    else:
        diffusion_trainer = pl.Trainer(
            devices=1,
            accelerator="gpu",
            precision=16,
            accumulate_grad_batches=4,
            callbacks=[ckpt_callback, exc_callback],
            log_every_n_steps=1,
            max_epochs=10000000,
            )
        
    if resume_ckpt is None:
        diffusion_trainer.fit(diffusion_model, dataloader)
    else:
        diffusion_trainer.fit(diffusion_model, dataloader, ckpt_path=resume_ckpt)
        
def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _path_exists(p):
  if p is not None:
    if os.path.exists(p):
      return p
    else:
      raise argparse.ArgumentTypeError('Input path does not exist.')
  return p


_examples = '''examples:'''


def main():
    parser = argparse.ArgumentParser(
        description = 'Train Encofusion Model on custom dataset',
        epilog=_examples, 
        formatter_class=argparse.RawDescriptionHelpFormatter
        )
    
    parser.add_argument('--train-data', help='Location of training data, MAKE SURE all files are .wav format and the same sample rate', required=True, metavar='DIR', type=_path_exists)
    parser.add_argument('--melody-data', help='Location of melody data, MAKE SURE all files are .wav format and the sampe sample rate', required=False,metavar='DIR', type=_path_exists)
    parser.add_argument('--text-data', help='Location of text data, MAKE SURE all files are .json format', required=False, metavar='DIR', type=_path_exists)
    parser.add_argument('--ckpt-save-location', help='Location to save network checkpoints', required=True, metavar='FILE', type=_path_exists)
    parser.add_argument('--log-to-wandb', help='T/F whether to log to weights and biases', default=True, metavar='BOOL', type=_str_to_bool)
    parser.add_argument('--resume-ckpt', help='Location of network pkl to resume training from', default=None, metavar='FILE', type=_path_exists)
    parser.add_argument('--num-workers', help='Number of workers dataloader should use, depends on machine, if you get a message about workers being a bottleneck, adjust to recommended size here', default=12, type=int)
    parser.add_argument('--batch-size', help='Number of batch_size', default=32, type=int)
    parser.add_argument('--embedding-weight', help='Conditioning embedding weight for demos', default=3.66, type=float)
    parser.add_argument('--ckpt-every', help='Number of training steps per checkpoint', default=5000, type=int)
    parser.add_argument('--project-name', help='Name of project', default='Encofusion', type=str)
    parser.add_argument('--use-cfg', help='whether using classifier free guidance', default=True, type=bool)
    args = parser.parse_args()
    run(**vars(args))

if __name__ == "__main__":
    main()