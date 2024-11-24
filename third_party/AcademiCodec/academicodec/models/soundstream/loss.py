import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram


def adversarial_g_loss(y_disc_gen):
    loss = 0.0
    for i in range(len(y_disc_gen)):
        #print(y_disc_gen[i].shape)
        # assert 1==2
        stft_loss = F.relu(1 - y_disc_gen[i]).mean().squeeze()
        loss += stft_loss
    return loss / len(y_disc_gen)


def feature_loss(fmap_r, fmap_gen):
    loss = 0.0
    for i in range(len(fmap_r)):
        for j in range(len(fmap_r[i])):
            stft_loss = ((fmap_r[i][j] - fmap_gen[i][j]).abs() /
                         (fmap_r[i][j].abs().mean())).mean()
            loss += stft_loss
    return loss / (len(fmap_r) * len(fmap_r[0]))


def sim_loss(y_disc_r, y_disc_gen):
    loss = 0.0
    for i in range(len(y_disc_r)):
        loss += F.mse_loss(y_disc_r[i], y_disc_gen[i])
    return loss / len(y_disc_r)


def sisnr_loss(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor, estimate value
          s: reference signal, N x S tensor, True value
    Return:
          sisnr: N tensor
    """
    if x.shape != s.shape:
        if x.shape[-1] > s.shape[-1]:
            x = x[:, :s.shape[-1]]
        else:
            s = s[:, :x.shape[-1]]

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError("Dimention mismatch when calculate si-snr, {} vs {}".
                           format(x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    loss = -20. * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))
    return torch.sum(loss) / x.shape[0]


def reconstruction_loss(x, G_x, args, eps=1e-7):
    L = 100 * F.mse_loss(x, G_x)  # wav L1 loss
    #loss_sisnr = sisnr_loss(G_x, x) # 
    #L += 0.01*loss_sisnr
    # print('L0 ', L)
    # print('loss_sisnr ', 0.01*loss_sisnr)
    # print('L0 ', L)
    for i in range(6, 11):
        s = 2**i
        melspec = MelSpectrogram(
            sample_rate=args.sr,
            n_fft=max(s, 512),
            win_length=s,
            hop_length=s // 4,
            n_mels=64,
            wkwargs={"device": args.device}).to(args.device)
        S_x = melspec(x)
        S_G_x = melspec(G_x)
        l1_loss = (S_x - S_G_x).abs().mean()
        l2_loss = (((torch.log(S_x.abs() + eps) - torch.log(S_G_x.abs() + eps))**2).mean(dim=-2)**0.5).mean()

        alpha = (s / 2) ** 0.5
        L += (l1_loss + alpha * l2_loss)
        #print('i ,loss ', i, loss)
    #assert 1==2
    return L


def criterion_d(y_disc_r, y_disc_gen, fmap_r_det, fmap_gen_det, y_df_hat_r,
                y_df_hat_g, fmap_f_r, fmap_f_g, y_ds_hat_r, y_ds_hat_g,
                fmap_s_r, fmap_s_g):
    loss = 0.0
    loss1 = 0.0
    loss2 = 0.0
    loss3 = 0.0
    loss_f = feature_loss(fmap_r_det, fmap_gen_det) + feature_loss(
        fmap_f_r, fmap_f_g) + feature_loss(fmap_s_r, fmap_s_g)
    for i in range(len(y_disc_r)):
        loss1 += F.relu(1 - y_disc_r[i]).mean() + F.relu(1 + y_disc_gen[
            i]).mean()
    for i in range(len(y_df_hat_r)):
        loss2 += F.relu(1 - y_df_hat_r[i]).mean() + F.relu(1 + y_df_hat_g[
            i]).mean()
    for i in range(len(y_ds_hat_r)):
        loss3 += F.relu(1 - y_ds_hat_r[i]).mean() + F.relu(1 + y_ds_hat_g[
            i]).mean()
    loss = (loss1 / len(y_disc_gen) + loss2 / len(y_df_hat_r) + loss3 /
            len(y_ds_hat_r)) / 3.0
    return loss + 0.0 * loss_f


def criterion_g(commit_loss, x, G_x, fmap_r, fmap_gen, y_disc_r, y_disc_gen,
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g, y_ds_hat_r,
                y_ds_hat_g, fmap_s_r, fmap_s_g, args):
    adv_g_loss = adversarial_g_loss(y_disc_gen)
    feat_loss = (feature_loss(fmap_r, fmap_gen) + sim_loss(
        y_disc_r, y_disc_gen) + feature_loss(fmap_f_r, fmap_f_g) + sim_loss(
            y_df_hat_r, y_df_hat_g) + feature_loss(fmap_s_r, fmap_s_g) +
                 sim_loss(y_ds_hat_r, y_ds_hat_g)) / 3.0
    rec_loss = reconstruction_loss(x.contiguous(), G_x.contiguous(), args)
    total_loss = args.LAMBDA_COM * commit_loss + args.LAMBDA_ADV * adv_g_loss + \
                 args.LAMBDA_FEAT * feat_loss + args.LAMBDA_REC * rec_loss
    return total_loss, adv_g_loss, feat_loss, rec_loss


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def adopt_dis_weight(weight, global_step, threshold=0, value=0.):
    if global_step % 3 == 0:  # 0,3,6,9,13....这些时间步，不更新dis
        weight = value
    return weight


def calculate_adaptive_weight(nll_loss, g_loss, last_layer, args):
    if last_layer is not None:
        nll_grads = torch.autograd.grad(
            nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    else:
        print('last_layer cannot be none')
        assert 1 == 2
    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 1.0, 1.0).detach()
    d_weight = d_weight * args.LAMBDA_ADV
    return d_weight


def loss_g(codebook_loss,
           inputs,
           reconstructions,
           fmap_r,
           fmap_gen,
           y_disc_r,
           y_disc_gen,
           global_step,
           y_df_hat_r,
           y_df_hat_g,
           y_ds_hat_r,
           y_ds_hat_g,
           fmap_f_r,
           fmap_f_g,
           fmap_s_r,
           fmap_s_g,
           last_layer=None,
           is_training=True,
           args=None):
    rec_loss = reconstruction_loss(inputs.contiguous(),
                                   reconstructions.contiguous(), args)
    adv_g_loss = adversarial_g_loss(y_disc_gen)
    adv_mpd_loss = adversarial_g_loss(y_df_hat_g)
    adv_msd_loss = adversarial_g_loss(y_ds_hat_g)
    adv_loss = (adv_g_loss + adv_mpd_loss + adv_msd_loss) / 3.0
    feat_loss = feature_loss(fmap_r, fmap_gen) + sim_loss(y_disc_r,
                                                          y_disc_gen)  # 
    feat_loss_mpd = feature_loss(fmap_f_r, fmap_f_g) + sim_loss(y_df_hat_r,
                                                                y_df_hat_g)
    feat_loss_msd = feature_loss(fmap_s_r, fmap_s_g) + sim_loss(y_ds_hat_r,
                                                                y_ds_hat_g)
    feat_loss_tot = (feat_loss + feat_loss_mpd + feat_loss_msd) / 3.0
    d_weight = torch.tensor(1.0)
    # try:
    #     d_weight = calculate_adaptive_weight(rec_loss, adv_g_loss, last_layer, args) # 动态调整重构损失和对抗损失
    # except RuntimeError:
    #     assert not is_training
    #     d_weight = torch.tensor(0.0)
    disc_factor = adopt_weight(
        args.LAMBDA_ADV, global_step, threshold=args.discriminator_iter_start)
    #feat_factor = adopt_weight(args.LAMBDA_FEAT, global_step, threshold=args.discriminator_iter_start)
    loss = rec_loss + d_weight * disc_factor * adv_loss + \
           args.LAMBDA_FEAT * feat_loss_tot + args.LAMBDA_COM * codebook_loss
    return loss, rec_loss, adv_loss, feat_loss_tot, d_weight


def loss_dis(y_disc_r_det, y_disc_gen_det, fmap_r_det, fmap_gen_det, y_df_hat_r,
             y_df_hat_g, fmap_f_r, fmap_f_g, y_ds_hat_r, y_ds_hat_g, fmap_s_r,
             fmap_s_g, global_step, args):
    disc_factor = adopt_weight(
        args.LAMBDA_ADV, global_step, threshold=args.discriminator_iter_start)
    d_loss = disc_factor * criterion_d(y_disc_r_det, y_disc_gen_det, fmap_r_det,
                                       fmap_gen_det, y_df_hat_r, y_df_hat_g,
                                       fmap_f_r, fmap_f_g, y_ds_hat_r,
                                       y_ds_hat_g, fmap_s_r, fmap_s_g)
    return d_loss
