import argparse
import itertools
import os
import time

import torch
import torch.distributed as dist
from academicodec.models.encodec.dataset import NSynthDataset
from academicodec.models.encodec.loss import criterion_d
from academicodec.models.encodec.loss import criterion_g
from academicodec.models.encodec.loss import loss_dis
from academicodec.models.encodec.loss import loss_g
from academicodec.models.encodec.msstftd import MultiScaleSTFTDiscriminator
from academicodec.models.encodec.net3 import SoundStream
from academicodec.models.soundstream.models import MultiPeriodDiscriminator
from academicodec.models.soundstream.models import MultiScaleDiscriminator
from academicodec.utils import Logger
from academicodec.utils import seed_everything
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--local_rank',
        default=-1,
        type=int,
        help='node rank for distributed training')
    # args for random
    parser.add_argument(
        '--seed',
        type=int,
        default=6666,
        help='seed for initializing training. ')
    parser.add_argument(
        '--cudnn_deterministic',
        action='store_true',
        help='set cudnn.deterministic True')
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='use tensorboard for logging')

    # args for training
    parser.add_argument(
        '--LAMBDA_WAV',
        type=float,
        default=100,
        help='hyper-parameter for wav time-domain loss')
    parser.add_argument(
        '--LAMBDA_ADV',
        type=float,
        default=1,
        help='hyper-parameter for adver loss')
    parser.add_argument(
        '--LAMBDA_FEAT',
        type=float,
        default=1,
        help='hyper-parameter for feat loss')
    parser.add_argument(
        '--LAMBDA_REC',
        type=float,
        default=1,
        help='hyper-parameter for rec loss')
    parser.add_argument(
        '--LAMBDA_COM',
        type=float,
        default=1000,
        help='hyper-parameter for commit loss')
    parser.add_argument(
        '--N_EPOCHS', type=int, default=100, help='Total training epoch')
    parser.add_argument(
        '--st_epoch', type=int, default=0, help='start training epoch')
    parser.add_argument(
        '--global_step', type=int, default=0, help='record the global step')
    parser.add_argument('--discriminator_iter_start', type=int, default=500)
    parser.add_argument('--BATCH_SIZE', type=int, default=10, help='batch size')
    parser.add_argument(
        '--PATH', type=str, default='model_path', help='model save path')
    parser.add_argument('--sr', type=int, default=16000, help='sample rate')
    parser.add_argument(
        '--print_freq', type=int, default=10, help='the print number')
    parser.add_argument(
        '--save_dir', type=str, default='log', help='log save path')
    parser.add_argument(
        '--train_data_path',
        type=str,
        # default='/apdcephfs_cq2/share_1297902/speech_user/shaunxliu/dongchao/code4/InstructTTS2/data_process/soundstream_data/train16k.lst', 
        default="/apdcephfs_cq2/share_1297902/speech_user/shaunxliu/data/codec_data_24k/train_valid_lists/train.lst",
        help='training data')
    parser.add_argument(
        '--valid_data_path',
        type=str,
        # default='/apdcephfs_cq2/share_1297902/speech_user/shaunxliu/dongchao/code4/InstructTTS2/data_process/soundstream_data/val16k.lst', 
        default="/apdcephfs_cq2/share_1297902/speech_user/shaunxliu/data/codec_data_24k/train_valid_lists/valid_256.lst",
        help='validation data')
    parser.add_argument(
        '--resume', action='store_true', help='whether re-train model')
    parser.add_argument(
        '--resume_path', type=str, default=None, help='resume_path')
    parser.add_argument(
        '--ratios',
        type=int,
        nargs='+',
        # probs(ratios) = hop_size
        default=[8, 5, 4, 2],
        help='ratios of SoundStream, shoud be set for different hop_size (32d, 320, 240d, ...)'
    )
    parser.add_argument(
        '--target_bandwidths',
        type=float,
        nargs='+',
        # default for 16k_320d
        default=[1, 1.5, 2, 4, 6, 12],
        help='target_bandwidths of net3.py')
    args = parser.parse_args()
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    if args.resume:
        args.PATH = args.resume_path  # direcly use the old model path
    else:
        args.PATH = os.path.join(args.PATH, time_str)
    args.save_dir = os.path.join(args.save_dir, time_str)
    os.makedirs(args.PATH, exist_ok=True)
    return args


def get_input(x):
    x = x.to(memory_format=torch.contiguous_format)
    return x.float()


def main():
    args = get_args()
    if args.seed is not None or args.cudnn_deterministic:
        seed_everything(args.seed, args.cudnn_deterministic)
    args.ngpus_per_node = torch.cuda.device_count()
    main_worker(args.local_rank, args)


def main_worker(local_rank, args):
    rank = local_rank
    args.local_rank = local_rank
    args.global_rank = local_rank
    args.distributed = args.ngpus_per_node > 1

    if args.ngpus_per_node > 1:
        from torch.distributed import init_process_group
        torch.cuda.set_device(local_rank)
        init_process_group(backend='nccl')

    #CUDA_VISIBLE_DEVICES = int(args.local_rank)
    logger = Logger(args)
    soundstream = SoundStream(
        n_filters=32, 
        D=512, 
        ratios=args.ratios,
        sample_rate=args.sr,
        target_bandwidths=args.target_bandwidths)
    msd = MultiScaleDiscriminator()
    mpd = MultiPeriodDiscriminator()
    stft_disc = MultiScaleSTFTDiscriminator(filters=32)

    if logger.is_primary:
        getModelSize(soundstream)
        getModelSize(msd)
        getModelSize(mpd)
        getModelSize(stft_disc)

    if args.distributed:
        soundstream = torch.nn.SyncBatchNorm.convert_sync_batchnorm(soundstream)
        stft_disc = torch.nn.SyncBatchNorm.convert_sync_batchnorm(stft_disc)
        msd = torch.nn.SyncBatchNorm.convert_sync_batchnorm(msd)
        mpd = torch.nn.SyncBatchNorm.convert_sync_batchnorm(mpd)

    # torch.distributed.barrier()
    args.device = torch.device('cuda', args.local_rank)
    soundstream.to(args.device)
    stft_disc.to(args.device)
    msd.to(args.device)
    mpd.to(args.device)
    find_unused_parameters = False
    if args.distributed:
        soundstream = DDP(
            soundstream,
            device_ids=[args.local_rank],
            find_unused_parameters=find_unused_parameters
        )  # device_ids=[args.local_rank], output_device=args.local_rank
        stft_disc = DDP(stft_disc,
                        device_ids=[args.local_rank],
                        find_unused_parameters=find_unused_parameters)
        msd = DDP(msd,
                  device_ids=[args.local_rank],
                  find_unused_parameters=find_unused_parameters)
        mpd = DDP(mpd,
                  device_ids=[args.local_rank],
                  find_unused_parameters=find_unused_parameters)
    # 这里之后需要看下 sr 的问题，如果输入 wav 的 sr 和 `--sr` 不一致则会有问题
    logger.log_info('Training set')
    train_dataset = NSynthDataset(audio_dir=args.train_data_path)
    logger.log_info('valid set')
    valid_dataset = NSynthDataset(audio_dir=args.valid_data_path)
    args.sr = train_dataset.sr
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, drop_last=True, shuffle=True)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dataset)
    else:
        train_sampler = None
        valid_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=8,
        sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=8,
        sampler=valid_sampler)
    logger.log_info("Build optimizers and lr-schedulers")
    optimizer_g = torch.optim.AdamW(
        soundstream.parameters(), lr=3e-4, betas=(0.5, 0.9))
    lr_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_g, gamma=0.999)
    optimizer_d = torch.optim.AdamW(
        itertools.chain(stft_disc.parameters(),
                        msd.parameters(), mpd.parameters()),
        lr=3e-4,
        betas=(0.5, 0.9))
    lr_scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_d, gamma=0.999)
    if args.resume:
        latest_info = torch.load(args.resume_path + '/latest.pth')
        args.st_epoch = latest_info['epoch']
        soundstream.load_state_dict(latest_info['soundstream'])
        stft_disc.load_state_dict(latest_info['stft_disc'])
        mpd.load_state_dict(latest_info['mpd'])
        msd.load_state_dict(latest_info['msd'])
        optimizer_g.load_state_dict(latest_info['optimizer_g'])
        lr_scheduler_g.load_state_dict(latest_info['lr_scheduler_g'])
        optimizer_d.load_state_dict(latest_info['optimizer_d'])
        lr_scheduler_d.load_state_dict(latest_info['lr_scheduler_d'])
    train(args, soundstream, stft_disc, msd, mpd, train_loader, valid_loader,
          optimizer_g, optimizer_d, lr_scheduler_g, lr_scheduler_d, logger)


def train(args, soundstream, stft_disc, msd, mpd, train_loader, valid_loader,
          optimizer_g, optimizer_d, lr_scheduler_g, lr_scheduler_d, logger):
    print('args ', args.global_rank)
    best_val_loss = float("inf")
    best_val_epoch = -1
    global_step = 0
    for epoch in range(args.st_epoch, args.N_EPOCHS + 1):
        soundstream.train()
        stft_disc.train()
        msd.train()
        mpd.train()
        train_loss_d = 0.0
        train_adv_g_loss = 0.0
        train_feat_loss = 0.0
        train_rec_loss = 0.0
        train_loss_g = 0.0
        train_commit_loss = 0.0
        k_iter = 0
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        for x in tqdm(train_loader):
            x = x.to(args.device)
            k_iter += 1
            global_step += 1  # record the global step
            for optimizer_idx in [0, 1]:  # we have two optimizer
                x_wav = get_input(x)
                G_x, commit_loss, last_layer = soundstream(x_wav)
                if optimizer_idx == 0:
                    # update generator
                    y_disc_r, fmap_r = stft_disc(x_wav.contiguous())
                    y_disc_gen, fmap_gen = stft_disc(G_x.contiguous())
                    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(
                        x_wav.contiguous(), G_x.contiguous())
                    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(
                        x_wav.contiguous(), G_x.contiguous())
                    total_loss_g, rec_loss, adv_g_loss, feat_loss, d_weight = loss_g(
                        commit_loss,
                        x_wav,
                        G_x,
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
                        last_layer=last_layer,
                        is_training=True,
                        args=args)
                    train_commit_loss += commit_loss.item()
                    train_loss_g += total_loss_g.item()
                    train_adv_g_loss += adv_g_loss.item()
                    train_feat_loss += feat_loss.item()
                    train_rec_loss += rec_loss.item()
                    optimizer_g.zero_grad()
                    total_loss_g.backward()
                    optimizer_g.step()
                else:
                    # update discriminator
                    y_disc_r_det, fmap_r_det = stft_disc(x.detach())
                    y_disc_gen_det, fmap_gen_det = stft_disc(G_x.detach())

                    # MPD
                    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(
                        x.detach(), G_x.detach())
                    #MSD
                    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(
                        x.detach(), G_x.detach())

                    loss_d = loss_dis(
                        y_disc_r_det, y_disc_gen_det, fmap_r_det, fmap_gen_det,
                        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g, y_ds_hat_r,
                        y_ds_hat_g, fmap_s_r, fmap_s_g, global_step, args)
                    train_loss_d += loss_d.item()
                    optimizer_d.zero_grad()
                    loss_d.backward()
                    optimizer_d.step()
            message = '<epoch:{:d}, iter:{:d}, total_loss_g:{:.4f}, adv_g_loss:{:.4f}, feat_loss:{:.4f}, rec_loss:{:.4f}, commit_loss:{:.4f}, loss_d:{:.4f}, d_weight: {:.4f}>'.format(
                epoch, k_iter,
                total_loss_g.item(),
                adv_g_loss.item(),
                feat_loss.item(),
                rec_loss.item(),
                commit_loss.item(), loss_d.item(), d_weight.item())
            if k_iter % args.print_freq == 0:
                logger.log_info(message)
        lr_scheduler_g.step()
        lr_scheduler_d.step()
        message = '<epoch:{:d}, <total_loss_g_train:{:.4f}, recon_loss_train:{:.4f}, adversarial_loss_train:{:.4f}, feature_loss_train:{:.4f}, commit_loss_train:{:.4f}>'.format(
            epoch, train_loss_g / len(train_loader), train_rec_loss /
            len(train_loader), train_adv_g_loss / len(train_loader),
            train_feat_loss / len(train_loader),
            train_commit_loss / len(train_loader))
        logger.log_info(message)
        with torch.no_grad():
            soundstream.eval()
            stft_disc.eval()
            mpd.eval()
            msd.eval()
            valid_loss_d = 0.0
            valid_loss_g = 0.0
            valid_commit_loss = 0.0
            valid_adv_g_loss = 0.0
            valid_feat_loss = 0.0
            valid_rec_loss = 0.0
            if args.distributed:
                valid_loader.sampler.set_epoch(epoch)
            for x in tqdm(valid_loader):
                x = x.to(args.device)
                for optimizer_idx in [0, 1]:
                    x_wav = get_input(x)
                    G_x, commit_loss, _ = soundstream(x_wav)
                    if optimizer_idx == 0:
                        valid_commit_loss += commit_loss
                        y_disc_r, fmap_r = stft_disc(x_wav.contiguous())
                        y_disc_gen, fmap_gen = stft_disc(G_x.contiguous())
                        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(
                            x_wav.contiguous(), G_x.contiguous())
                        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(
                            x_wav.contiguous(), G_x.contiguous())

                        total_loss_g, adv_g_loss, feat_loss, rec_loss = criterion_g(
                            commit_loss,
                            x_wav,
                            G_x,
                            fmap_r,
                            fmap_gen,
                            y_disc_r,
                            y_disc_gen,
                            y_df_hat_r,
                            y_df_hat_g,
                            fmap_f_r,
                            fmap_f_g,
                            y_ds_hat_r,
                            y_ds_hat_g,
                            fmap_s_r,
                            fmap_s_g,
                            args=args)
                        valid_loss_g += total_loss_g.item()
                        valid_adv_g_loss += adv_g_loss.item()
                        valid_feat_loss += feat_loss.item()
                        valid_rec_loss += rec_loss.item()
                    else:
                        y_disc_r_det, fmap_r_det = stft_disc(
                            x_wav.contiguous().detach())
                        y_disc_gen_det, fmap_gen_det = stft_disc(
                            G_x.contiguous().detach())
                        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(
                            x_wav.contiguous().detach(),
                            G_x.contiguous().detach())
                        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(
                            x_wav.contiguous().detach(),
                            G_x.contiguous().detach())
                        loss_d = criterion_d(y_disc_r_det, y_disc_gen_det,
                                             fmap_r_det, fmap_gen_det,
                                             y_df_hat_r, y_df_hat_g, fmap_f_r,
                                             fmap_f_g, y_ds_hat_r, y_ds_hat_g,
                                             fmap_s_r, fmap_s_g)
                        valid_loss_d += loss_d.item()
            if dist.get_rank() == 0:
                best_model = soundstream.state_dict().copy()
                latest_model_soundstream = soundstream.state_dict().copy()
                latest_model_dis = stft_disc.state_dict().copy()
                latest_mpd = mpd.state_dict().copy()
                latest_msd = msd.state_dict().copy()
                if valid_rec_loss < best_val_loss:
                    best_val_loss = valid_rec_loss
                    best_val_epoch = epoch
                torch.save(best_model,
                           args.PATH + '/best_' + str(epoch) + '.pth')
                latest_save = {}
                latest_save['soundstream'] = latest_model_soundstream
                latest_save['stft_disc'] = latest_model_dis
                latest_save['mpd'] = latest_mpd
                latest_save['msd'] = latest_msd
                latest_save['epoch'] = epoch
                latest_save['optimizer_g'] = optimizer_g.state_dict()
                latest_save['optimizer_d'] = optimizer_d.state_dict()
                latest_save['lr_scheduler_g'] = lr_scheduler_g.state_dict()
                latest_save['lr_scheduler_d'] = lr_scheduler_d.state_dict()
                torch.save(latest_save, args.PATH + '/latest.pth')

            message = '<epoch:{:d}, total_loss_g_valid:{:.4f}, recon_loss_valid:{:.4f}, adversarial_loss_valid:{:.4f}, feature_loss_valid:{:.4f}, commit_loss_valid:{:.4f}, valid_loss_d:{:.4f}, best_epoch:{:d}>'.format(
                epoch, valid_loss_g / len(valid_loader), valid_rec_loss /
                len(valid_loader), valid_adv_g_loss / len(valid_loader),
                valid_feat_loss / len(valid_loader),
                valid_commit_loss / len(valid_loader),
                valid_loss_d / len(valid_loader), best_val_epoch)
            logger.log_info(message)


if __name__ == '__main__':
    main()
