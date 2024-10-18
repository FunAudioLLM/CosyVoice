from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from matcha.hifigan.models import feature_loss, generator_loss, discriminator_loss
from cosyvoice.utils.losses import tpr_loss, mel_loss


class HiFiGan(nn.Module):
    def __init__(self, generator, discriminator, mel_spec_transform,
                 multi_mel_spectral_recon_loss_weight=45, feat_match_loss_weight=2.0,
                 tpr_loss_weight=1.0, tpr_loss_tau=0.04):
        super(HiFiGan, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.mel_spec_transform = mel_spec_transform
        self.multi_mel_spectral_recon_loss_weight = multi_mel_spectral_recon_loss_weight
        self.feat_match_loss_weight = feat_match_loss_weight
        self.tpr_loss_weight = tpr_loss_weight
        self.tpr_loss_tau = tpr_loss_tau

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if batch['turn'] == 'generator':
            return self.forward_generator(batch, device)
        else:
            return self.forward_discriminator(batch, device)

    def forward_generator(self, batch, device):
        real_speech = batch['speech'].to(device)
        pitch_feat = batch['pitch_feat'].to(device)
        # 1. calculate generator outputs
        generated_speech, generated_f0 = self.generator(batch, device)
        # 2. calculate discriminator outputs
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.discriminator(real_speech, generated_speech)
        # 3. calculate generator losses, feature loss, mel loss, tpr losses [Optional]
        loss_gen, _ = generator_loss(y_d_gs)
        loss_fm = feature_loss(fmap_rs, fmap_gs)
        loss_mel = mel_loss(real_speech, generated_speech, self.mel_spec_transform)
        if self.tpr_loss_weight != 0:
            loss_tpr = tpr_loss(y_d_rs, y_d_gs, self.tpr_loss_tau)
        else:
            loss_tpr = torch.zeros(1).to(device)
        loss_f0 = F.l1_loss(generated_f0, pitch_feat)
        loss = loss_gen + self.feat_match_loss_weight * loss_fm + \
            self.multi_mel_spectral_recon_loss_weight * loss_mel + \
            self.tpr_loss_weight * loss_tpr + loss_f0
        return {'loss': loss, 'loss_gen': loss_gen, 'loss_fm': loss_fm, 'loss_mel': loss_mel, 'loss_tpr': loss_tpr, 'loss_f0': loss_f0}

    def forward_discriminator(self, batch, device):
        real_speech = batch['speech'].to(device)
        # 1. calculate generator outputs
        with torch.no_grad():
            generated_speech, generated_f0 = self.generator(batch, device)
        # 2. calculate discriminator outputs
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.discriminator(real_speech, generated_speech)
        # 3. calculate discriminator losses, tpr losses [Optional]
        loss_disc, _, _ = discriminator_loss(y_d_rs, y_d_gs)
        if self.tpr_loss_weight != 0:
            loss_tpr = tpr_loss(y_d_rs, y_d_gs, self.tpr_loss_tau)
        else:
            loss_tpr = torch.zeros(1).to(device)
        loss = loss_disc + self.tpr_loss_weight * loss_tpr
        return {'loss': loss, 'loss_disc': loss_disc, 'loss_tpr': loss_tpr}
