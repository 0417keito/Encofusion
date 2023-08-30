import os
import torch
import argparse
from omegaconf import OmegaConf
from utils.utils import read_yaml_file, parse_diff_conf, preprocess_audio, get_base_noise, save_audio
from utils.audio_diffusion import AudioDiffusion
from audiocraft.models import builders


device = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG_FILE = './configs/audio_diffusion/sampling.yaml'

def run(*args, **kwargs):
    conf = read_yaml_file(CONFIG_FILE)
    text = kwargs["text"]
    melody = kwargs["melody"]
    save_dir = kwargs["save_dir"]
    batch_size = kwargs["batch_size"]
    seconds_length = kwargs["seconds_length"]
    init_audio = kwargs["init_audio"]
    init_strength = kwargs["init_strength"]
    noise_seed = kwargs["noise_seed"]
    use_cfg = kwargs["use_cfg"]
    num_steps = kwargs["num_steps"]
    embedding_strength = kwargs["embedding_strength"]
    
    diffusion_conf = conf["model"]["diffusion"]
    diffusion_conf = parse_diff_conf(diffusion_conf)
    
    encodec_conf = OmegaConf.load(conf["model"]["lm"]["conf_loc"])
    encodec_ckpt = conf["model"]["encodec"]["ckpt_loc"]
    
    lm_conf = OmegaConf.load(conf["model"]["lm"]["conf_loc"])
    lm_ckpt = conf["model"]["lm"]["ckpt_loc"]
    
    compression_model = builders.get_compressino_model(encodec_conf)
    compression_model.load_state_dict(torch.load(encodec_ckpt)["best_state"])
    
    lm = builders.get_lm_model(lm_conf)
    lm.load_state_dict(torch.load(lm_ckpt)["best_state"])
    sr = compression_model.sample_rate
    
    diffusion_model = AudioDiffusion(compression_model, lm, diffusion_kwargs=diffusion_conf,
                                     use_cfg=use_cfg, device=device)
    diffusion_model.load_state_dict(torch.load(conf["model"]["diffusion"]["ckpt_loc"]))
    diffusion_model = diffusion_model.requires_grad_(False).to(device)
    diffusion_model.eval()
    assert not diffusion_model.diffusion.training
    assert not diffusion_model.lm.training
    assert not diffusion_model.compression_model.training

    noise = get_base_noise(noise_seed=noise_seed, model=compression_model, 
                           batch_size=batch_size, sample_rate=sr, duration = seconds_length)
    
    if init_audio is not None:
        init_enc = preprocess_audio(init_audio, compression_model, duration=seconds_length)
        sample, sample_audio = diffusion_model.sample(noise=noise,
                                                      num_steps=num_steps,
                                                      init=init_enc,
                                                      init_strength=init_strength,
                                                      context_prompt=text,
                                                      context_melody=melody,
                                                      context_strength=embedding_strength)
        
    sample, sample_audio = diffusion_model.sample(noise=noise,
                                                  num_steps=num_steps,
                                                  context_prompt=text,
                                                  context_melody=melody,
                                                  context_strength=embedding_strength)
    
    save_audio(audio=sample_audio, save_dir=save_dir, sr=sr)
    
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
        description = 'Sample from Encofusion', 
        epilog=_examples, 
        formatter_class=argparse.RawDescriptionHelpFormatter
        )
    parser.add_argument('--text', help='text using text-condition-sampling',required=True, type=str)
    parser.add_argument('--melody', help='Optionally provide location of audio using melody conditioning', default=None, metavar='FILE', type=_path_exists)
    parser.add_argument('--save-dir', help='Name of directory for saved files', required=True, type=str)
    parser.add_srgument('--batch-size', help='Number of batch_size', default=32, type=int)
    parser.add_argument('--seconds-length', help='Length in seconds of sampled audio', default=30, type=int)
    parser.add_argument('--init-audio', help='Optionally provide location of init audio to alter using diffusion', default=None, metavar='FILE', type=_path_exists)
    parser.add_argument('--init-strength', help='The init strength alters the range of time conditioned steps used to diffuse init audio, float between 0-1, 1==return original image, 0==diffuse from noise', default=0.0, type=float)
    parser.add_argument('--noise-seed', help='Random seed to use for sampling base layer of Jukebox Diffusion', default=None, type=int)
    parser.add_argument('--use-cfg', help='Whether using classifier free guidance', default=True, type=bool)
    parser.add_argument('--num_steps', help='Number of steps', default=100, type=int)
    parser.add_argument('--embedding-strength', help='The strength of conditioning', default=1, default=0.0, type=float)
    args = parser.parse_args()
    
    run(**vars(args))

if __name__ == "__main__":
    main()
