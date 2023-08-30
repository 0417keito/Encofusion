import yaml 
import importlib
import sys
import os
import pytorch_lightning as pl
import torchaudio
import random
import soundfile as sf
import torch
import typing as tp
from datetime import datetime
from torch.utils.data import Dataset
from audiocraft.models.musicgen import MelodyList
from audiocraft.modules.conditioners import ConditioningAttributes, WavCondition
from audiocraft.models.encodec import CompressionModel
from audiocraft.models.lm import LMModel


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
    return data

def save_audio(audio, save_dir, sr):
    now = datetime.now()
    formatted_now = now.strftime("%m-%d-%H:%M")
    file = f'{save_dir}/{formatted_now}/result.wav'
    sf.write(file, audio, sr)

def parse_diff_conf(diff_conf):
    new_conf = {k:(get_obj_from_str(v) if '_t' in k else v) for k,v in diff_conf.items()}
    return new_conf

def load_audio(audio_path, model:CompressionModel, duration:int=30):
    wav, sr = torchaudio.load(audio_path)
    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
    wav = wav.mean(dim=0, keepdim=True)
    
    target_sample_length = int(model.sample_rate * duration)
    current_sample_length = wav.shape[1]
    if current_sample_length > target_sample_length:
        end_sample = target_sample_length
        start_sample = random.randrange(0, max(current_sample_length - end_sample, 1))
        wav = wav[:, start_sample:start_sample+end_sample]
    elif current_sample_length < target_sample_length:
        pad_amount = target_sample_length - current_sample_length
        wav = torch.cat([wav, torch.zeros(1, pad_amount)], dim=1)
        
    assert wav.shape[0] == 1
    wav = wav.cuda()
    wav = wav.unsqueeze(1)
    
    return wav

def preprocess_audio(audio_path, model:CompressionModel, duration:int=30):
    wav, sr = torchaudio.load(audio_path)
    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
    wav = wav.mean(dim=0, keepdim=True)
    
    target_sample_length = int(model.sample_rate * duration)
    current_sample_length = wav.shape[1]
    if current_sample_length > target_sample_length:
        end_sample = target_sample_length
        start_sample = random.randrange(0, max(current_sample_length - end_sample, 1))
        wav = wav[:, start_sample:start_sample+end_sample]
    elif current_sample_length < target_sample_length:
        pad_amount = target_sample_length - current_sample_length
        wav = torch.cat([wav, torch.zeros(1, pad_amount)], dim=1)
        
    assert wav.shape[0] == 1
    
    wav = wav.cuda()
    wav = wav.unsqueeze(1)
    code, scale = model.encode(wav)
    emb = model.decode_latent(code)

    return emb, code

def postprocess_audio(embs, model:CompressionModel, scale=None):
    out = model.decoder(embs)
    out = model.postprocess(out, scale)
    
    return out

def preprocess_melody(melody_path, model:CompressionModel, duration=30):
    melody, sr = torchaudio.load(melody_path)
    melody = torchaudio.functional.resample(melody, sr, model.sample_rate)
    melody = melody.mean(dim=0, keepdim=True)
    
    target_sample_length = int(model.sample_rate * duration)
    current_sample_length = melody.shape[1]
    if current_sample_length > target_sample_length:
        end_sample = target_sample_length
        start_sample = random.randrange(0, max(current_sample_length - end_sample, 1))
        melody = melody[:, start_sample:start_sample+end_sample]
    elif current_sample_length < target_sample_length:
        pad_amount = target_sample_length - current_sample_length
        melody = torch.cat([melody, torch.zeros(1, pad_amount)], dim=1)
        
    assert melody.shape[0] == 1
    
    melody = melody.cuda()
    
    return melody

def extend_dim(x: torch.Tensor, dim: int):
    # e.g. if dim = 4: shape [b] => [b, 1, 1, 1],
    return x.view(*x.shape + (1,) * (dim - x.ndim))

def custom_random_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

def get_base_noise(noise_seed, model:CompressionModel, batch_size,
                   sample_rate:int, duration:int = 30):
    rng = custom_random_generator(noise_seed)
    wav_length = sample_rate * duration
    wav = torch.randn(batch_size, 1, wav_length, generator=rng)
    wav = wav.cuda()
    code, _ = model.encode(wav)
    noise = model.decode_latent(code)
    return noise
    
def prepare_tokens_and_attributes(
    compression_model:CompressionModel, 
    lm:LMModel,
    descriptions: tp.Sequence[tp.Optional[str]],
    prompt: tp.Optional[torch.Tensor],
    melody_wavs: tp.Optional[MelodyList] = None,
    device = 'cpu',
    ) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[torch.Tensor]]:
        """Prepare model inputs.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            prompt (torch.Tensor): A batch of waveforms used for continuation.
            melody_wavs (torch.Tensor, optional): A batch of waveforms
                used as melody conditioning. Defaults to None.
        """
        attributes = [
            ConditioningAttributes(text={'description': description})
            for description in descriptions]

        if melody_wavs is None:
            for attr in attributes:
                attr.wav['self_wav'] = WavCondition(
                    torch.zeros((1, 1, 1), device=device),
                    torch.tensor([0], device=device),
                    sample_rate=[compression_model.sample_rate],
                    path=[None])
        else:
            if 'self_wav' not in lm.condition_provider.conditioners:
                raise RuntimeError("This model doesn't support melody conditioning. "
                                   "Use the `melody` model.")
            assert len(melody_wavs) == len(descriptions), \
                f"number of melody wavs must match number of descriptions! " \
                f"got melody len={len(melody_wavs)}, and descriptions len={len(descriptions)}"
            for attr, melody in zip(attributes, melody_wavs):
                if melody is None:
                    attr.wav['self_wav'] = WavCondition(
                        torch.zeros((1, 1, 1), device=device),
                        torch.tensor([0], device=device),
                        sample_rate=[compression_model.sample_rate],
                        path=[None])
                else:
                    attr.wav['self_wav'] = WavCondition(
                        melody[None].to(device=device),
                        torch.tensor([melody.shape[-1]], device=device),
                        sample_rate=[compression_model.sample_rate],
                        path=[None],
                    )

        if prompt is not None:
            if descriptions is not None:
                assert len(descriptions) == len(prompt), "Prompt and nb. descriptions doesn't match"
            prompt = prompt.to(device)
            prompt_tokens, scale = compression_model.encode(prompt)
            assert scale is None
        else:
            prompt_tokens = None
        return attributes, prompt_tokens

def my_collate(batch):
    all_audio = [item[0] for item in batch]
    all_label = [item[1] for item in batch]
    all_melody = [item[2] for item in batch]
    return [(audio, label, melody) for audio, label, melody in zip(all_audio, all_label, all_melody)]

class AudioDataset(Dataset):
    def __init__(self, audio_data, text_data, melody_data=None):
        self.audio_data = audio_data
        self.text_data = text_data
        self.melody_data = melody_data
        self.data_map = []
        dir_map = os.listdir(audio_data)
        
        for d in dir_map:
            name, ext = os.path.splitext(d)
            if ext == ".wav":
                text_path = os.path.join(text_data, name + ".json")
                melody_path = os.path.join(melody_data, name + ".wav") if melody_data else None
                
                if os.path.exists(text_path):
                    item = {
                        "audio": os.path.join(audio_data, d),
                        "label": text_path
                    }
                    if melody_data and os.path.exists(melody_path):
                        item["melody"] = melody_path
                    self.data_map.append(item)
                               
    def __len__(self):
        return len(self.data_map)
    
    def __getitem__(self, idx):
        data = self.data_map[idx]
        audio = data["audio"]
        label = data.get("label", "")
        melody = data.get("melody", "")
        return audio, label, melody

class DemoCallback(pl.Callback):
    '''
    Class for demoing during training

    Init Params
    ____________
    - global_args: (DemoArgs class) kwargs for demoing
    '''
    def __init__(self, global_args):
        super().__init__()
        self.demo_every = global_args.demo_every
        self.num_demos = global_args.num_demos
        self.demo_samples = global_args.sample_size
        self.demo_steps = global_args.demo_steps
        self.sample_rate = global_args.sample_rate
        self.last_demo_step = -1
        self.base_samples = global_args.base_samples
        self.base_tokens = global_args.base_tokens
        self.dirpath = global_args.dirpath
        self.embedding_scale = global_args.embedding_scale
        self.context_mult = global_args.context_mult

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)