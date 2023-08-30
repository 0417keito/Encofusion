import pytorch_lightning as pl
from flat_audio_diffusion.models import DiffusionModel
from utils.utils import (preprocess_audio, postprocess_audio, prepare_tokens_and_attributes, 
                         extend_dim, preprocess_melody)
import torch
import tqdm
from einops import repeat
from audiocraft.models.encodec import CompressionModel
from audiocraft.models.lm import LMModel
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout

device = "cuda" if torch.cuda.is_available() else "cpu"

class AudioDiffusion(pl.LightningModule):
    def __init__(self, compression_model:CompressionModel, lm:LMModel, use_cfg, diffusion_kwargs):
        super().__init__()
        self.diffusion = DiffusionModel(**diffusion_kwargs)
        self.compression_model = compression_model
        self.lm = lm
        self.use_cfg = use_cfg
        self.compression_model.eval()
        self.lm.eval()
        
    def configure_optimizers(self):
        return torch.optim.Adam([self.diffusion.parameters()], lr=4e-5)
    
    def training_step(self, batch, batch_idx):
        self.diffusion.train()
        assert self.diffusion.training
        
        all_embs = []
        all_texts = []
        all_melody = []
        for inner_audio, cond, melody in batch:
            emb, _ = preprocess_audio(inner_audio, self.compression_model)
            if inner_audio is None: continue
            if self.use_cfg:
                embs = torch.cat([emb, emb], dim=0)
            else:
                embs = embs
                
            if melody is not None:
                melody = preprocess_melody(melody, self.compression_model)
            
            all_embs.append(embs)
            all_texts.append(open(cond, "r").read().strip())
            all_melody.append(melody)
        
        if not all_melody:
            attributes, _ = prepare_tokens_and_attributes(compression_model=self.compression_model,
                                                        lm=self.lm, descriptions=all_texts,prompt=None,
                                                        melody_wavs=None, device=device)
        else:
            attributes, _ = prepare_tokens_and_attributes(compression_model=self.compression_model,
                                                        lm=self.lm, descriptions=all_texts,prompt=None,
                                                        melody_wavs=all_melody, device=device)
        conds = attributes
        if self.use_cfg:
            null_conds = ClassifierFreeGuidanceDropout(p=1.0)(conds)
            conds = conds + null_conds
        tokenized = self.lm.condition_provider.tokenize(conds)
        cfg_conditions = self.lm.condition_provider(tokenized)
        condition_tensors = cfg_conditions
        
        wav_condition, wav_mask = condition_tensors["self_wav"]
        text_condition, text_mask = condition_tensors["description"]
        
        conds = torch.cat([wav_condition, text_condition], dim=1)
        embs = torch.cat(all_embs, dim=0)
        
        loss = self.diffusion(embs, embedding=conds, embedding_mask_proba=0.1)
        
        log_dict = {
            "train/loss": loss.detach()
        }
        self.log_dict(log_dict, prog_bar=True, on_step=True)
        
        return loss
    
    def sample(self, noise, num_steps, init, init_strength, context_prompt,
               context_melody, context_strength):
        
        if context_prompt is not None:
            text = list(context_prompt)
            
        if context_melody is not None:
            context_melody = preprocess_melody(context_melody, self.compression_model)
            melody = list(context_melody)
            attributes, _ = prepare_tokens_and_attributes(compression_model=self.compression_model,
                                                          lm=self.lm, descriptions=text,
                                                          melody_wavs=melody, device=device)
        else:
            attributes, _ = prepare_tokens_and_attributes(compression_model=self.compression_model,
                                                          lm=self.lm, descriptions=text,
                                                          melody_wavs=None, device=device)
        conds = attributes
            
        if self.use_cfg:
            null_conds = ClassifierFreeGuidanceDropout(p=1.0)(conds)
            conds = conds + null_conds
        tokenized = self.lm.condition_provider.tokenize(conds)
        cfg_conditions = self.lm.condition_provider(tokenized)
        condition_tensors = cfg_conditions

        wav_condition, wav_mask = condition_tensors["self_wav"]
        text_condition, text_mask = condition_tensors["description"]
        conds = torch.cat(wav_condition, text_condition, dim=1)
        
        if init is not None:
            start_step = int(init_strength*num_steps)
            sigmas = self.diffusion.sampler.schedule(num_steps + 1, device=device)
            sigmas = sigmas[start_step:]
            sigmas = repeat(sigmas, "i -> i b", b=1)
            sigmas_batch = extend_dim(sigmas, dim=noise.ndim + 1)
            alphas, betas = self.diffusion.sampler.get_alpha_beta(sigmas_batch)
            alpha, beta = alphas[0], betas[0]
            x_noisy = alpha*init + beta*noise
            progress_bar = tqdm.tqdm(range(num_steps-start_step), disable=False)

            for i in progress_bar:
                v_pred = self.diffusion.sampler.net(x_noisy, sigmas[i], embedding=conds, embedding_scale=context_strength)
                x_pred = alphas[i] * x_noisy - betas[i] * v_pred
                noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
                x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred #sample
                progress_bar.set_description(f"Sampling (noise={sigmas[i+1,0]:.2f})")
                
                x_noisy_audio = postprocess_audio(x_noisy, self.compression_model)
                
                return x_noisy, x_noisy_audio
        else:
            sample = self.diffusion.sample(
                    noise,
                    embedding=conds,
                    embedding_scale=context_strength, 
                    num_steps=num_steps
                    )
            
            sample_audio = postprocess_audio(sample, self.compression_model)

            return sample, sample_audio
            