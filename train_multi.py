import os
import time
import random
import argparse
import logging
import footsteps
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import  DataLoader
from torch.amp import autocast, GradScaler
import wandb

import dataset_multi
from icon_registration.losses import to_floats_generic, BASICLoss, ICONLoss, ICONLoss_can, ICONLoss_can_unsupervised, ICONLoss_can_mono, ICONLoss_can_mono_unsupervised, MINDSSC
from models import make_network

def minmax_norm(img):
    return (img - img.min()) / (img.max() - img.min())

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")

    # 기존 핸들러 제거
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 로그 설정
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S",)

    # 터미널에도 로그 출력 (콘솔 핸들러 추가)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logging.getLogger().addHandler(console_handler)

def write_stats(args, stats, step, prefix=""):
    log_data = {}
    for k, v in to_floats_generic(args, stats)._asdict().items():
        log_data[f"{prefix}{k}"] = v

    # if args.training_type == 'basic':
    #     for k, v in to_floats(stats)._asdict().items():
    #         log_data[f"{prefix}{k}"] = v
    # elif args.training_type == 'unsupervised':
    #     for k, v in to_floats_can_unsupervised(args, stats)._asdict().items():
    #         log_data[f"{prefix}{k}"] = v
    # else:
    #     for k, v in to_floats_can(args, stats)._asdict().items():
    #         log_data[f"{prefix}{k}"] = v
    wandb.log(log_data, step=step)

def augment(img_A, img_B, A_type="image", B_type="image", separately=False):
    device = img_A.device
    batch_size = img_A.shape[0]

    identity = torch.eye(3, 4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    noise = torch.randn((batch_size, 3, 4), device=device) * 0.05
    forward = identity + noise

    grid_shape = list(img_A.shape)
    forward_grid = F.affine_grid(forward, grid_shape, align_corners=False)

    forward_grid2 = forward_grid
    if separately:
        noise2 = torch.randn((batch_size, 3, 4), device=device) * 0.05
        forward2 = identity + noise2
        forward_grid2 = F.affine_grid(forward2, grid_shape, align_corners=False)

    if A_type == "image":
        warped_img_A = F.grid_sample(img_A, forward_grid, padding_mode='border')
    else:
        warped_img_A = F.grid_sample(img_A.float(), forward_grid, mode='nearest', padding_mode='border')

    if B_type == "image":
        warped_img_B = F.grid_sample(img_B, forward_grid2, padding_mode='border')
    else:
        warped_img_B = F.grid_sample(img_B.float(), forward_grid2, mode='nearest', padding_mode='border')

    return warped_img_A, warped_img_B

def train_kernel(args, optimizer, scaler, net, moving_image, fixed_image, moving_label, fixed_label, moving_cano, fixed_cano, ite):
    optimizer.zero_grad()
    assert net.training, "Network accidentally left in eval() mode!"

    # Use mixed precision for the forward and loss computation
    with autocast(device_type = 'cuda'):
        loss_object = net(moving_image, fixed_image, 
                          moving_cano, fixed_cano, 
                        #   moving_label, fixed_label
                          )
        loss = torch.mean(loss_object.all_loss)
        assert loss.requires_grad, "❌ Loss tensor has no gradients — check loss construction."

    # Backward pass with scaled loss
    scaler.scale(loss).backward()
    logging.info(f"Grad mean: {torch.cat([p.grad.view(-1) for p in net.parameters() if p.grad is not None]).mean().item()}")


    # Optimizer step (with gradient scaling)
    if scaler.get_scale() == 1.0:
        logging.info("⚠️ GradScaler scale reset — possible NaN or inf gradients.")
    scaler.step(optimizer)
    scaler.update()

    # Logging
    logging.info(to_floats_generic(args, loss_object))
    # if args.training_type == 'basic':
    #     logging.info(to_floats_basic(loss_object))
    #     # logging.info(to_floats(loss_object))
    # elif args.training_type == 'unsupervised':
    #     logging.info(to_floats_can_unsupervised(args, loss_object))
    # else:
    #     logging.info(to_floats_can(args, loss_object))

    write_stats(args, loss_object, ite, prefix="train/")

    # # --- Save intermediate transformed outputs as NIfTI ---
    # import nibabel as nib
    # import os

    # # Determine actual network object
    # net_ref = net
    # if isinstance(net, torch.nn.DataParallel):
    #     net_ref = net.module

    # # Extract images
    # warped_T2 = net_ref.warped_image_A.detach().cpu().squeeze().numpy()
    # moving_T2 = moving_image.detach().cpu().squeeze().numpy()
    # fixed_b0 = fixed_image.detach().cpu().squeeze().numpy()
    # print(np.linalg.norm(moving_T2-warped_T2))

    # # Extract canonical outputs if available
    # try:
    #     warped_T2_cano = net_ref.warped_cano_A.detach().cpu().squeeze().numpy()
    # except AttributeError:
    #     warped_T2_cano = None

    # # Extract segmentations if provided
    # try:
    #     moving_seg = moving_label.detach().cpu().squeeze().numpy()
    # except AttributeError:
    #     moving_seg = None

    # try:
    #     fixed_seg = fixed_label.detach().cpu().squeeze().numpy()
    # except AttributeError:
    #     fixed_seg = None

    # # Save
    # save_dir = os.path.join("./tmp_nifti")
    # os.makedirs(save_dir, exist_ok=True)
    # iter_tag = f"iter_{ite:06d}"

    # # Original images
    # nib.save(nib.Nifti1Image(moving_T2, np.eye(4)), os.path.join(save_dir, f"{iter_tag}_T2_orig.nii.gz"))
    # nib.save(nib.Nifti1Image(fixed_b0, np.eye(4)), os.path.join(save_dir, f"{iter_tag}_b0_orig.nii.gz"))

    # # Warped / transformed outputs
    # nib.save(nib.Nifti1Image(warped_T2, np.eye(4)), os.path.join(save_dir, f"{iter_tag}_T2_reg.nii.gz"))
    # if warped_T2_cano is not None:
    #     nib.save(nib.Nifti1Image(warped_T2_cano, np.eye(4)), os.path.join(save_dir, f"{iter_tag}_T2_cano.nii.gz"))

    # # Segmentations
    # if moving_seg is not None:
    #     nib.save(nib.Nifti1Image(moving_seg.astype(np.uint8), np.eye(4)), os.path.join(save_dir, f"{iter_tag}_T2_seg.nii.gz"))
    # if fixed_seg is not None:
    #     nib.save(nib.Nifti1Image(fixed_seg.astype(np.uint8), np.eye(4)), os.path.join(save_dir, f"{iter_tag}_b0_seg.nii.gz"))


def train(args,
    net,
    optimizer,
    train_loader,
    val_dataset,
    step_callback=(lambda net: None),
    unwrapped_net=None
):
    """A training function intended for long running experiments, with tensorboard logging
    and model checkpoints. Use for medical registration training
    """

    if unwrapped_net is None:
        unwrapped_net = net

    iteration = 0
    start_epoch = 0
    if args.resume_from != "":
        iteration = int(args.resume_from.split("_")[-1][4:])
        start_epoch = int(args.resume_from.split("_")[-2][1:])

    total_start_time = time.time()
    for epoch in tqdm(range(start_epoch, args.epoch)):
        epoch_start_time = time.time()
        train_start_time = time.time()
        scaler = GradScaler(device = 'cuda')
        for batch_iter in train_loader:
            if args.training_type=='basic':
                moving_image, fixed_image = batch_iter[0].cuda(), batch_iter[1].cuda()
                moving_label, fixed_label, moving_cano, fixed_cano = None, None, None, None

            else:
                moving_image, fixed_image, moving_label, fixed_label, moving_cano, fixed_cano = batch_iter
                moving_image, fixed_image, moving_label, fixed_label = moving_image.cuda(), fixed_image.cuda(), moving_label.cuda(), fixed_label.cuda()
                moving_cano, fixed_cano = moving_cano.cuda(), fixed_cano.cuda()
            # import matplotlib.pyplot as plt
            # os.makedirs('./temp', exist_ok = True)
            # z = fixed_image.shape[-1]//2
            # fig, ax = plt.subplots(2, 2)
            # ax=ax.flatten()
            # ax[0].imshow(fixed_image.cpu().numpy()[0, 0, :,:,z], cmap = plt.cm.Greys_r)
            # ax[1].imshow(fixed_label.cpu().numpy()[0, 0, :,:,z], cmap = plt.cm.Greys_r)
            # ax[2].imshow(moving_image.cpu().numpy()[0, 0, :,:,z], cmap = plt.cm.Greys_r)
            # ax[3].imshow(moving_label.cpu().numpy()[0, 0, :,:,z], cmap = plt.cm.Greys_r)
            # plt.savefig('./temp/test.png', bbox_inches='tight', dpi = 300)

            if args.augment:
                with torch.no_grad():
                    moving_image, moving_label = augment(moving_image, moving_label, A_type="image", B_type="label")
                    fixed_image, fixed_label = augment(fixed_image, fixed_label, A_type="image", B_type="label")
                    # if args.num_cano != '0':
                    #     if args.num_cano == '-1':
                    moving_cano, fixed_cano = augment(moving_cano, fixed_cano, A_type="image", B_type="image", separately=True)
                        # else:
                        #     moving_cano, fixed_cano = augment(moving_cano, fixed_cano, A_type="image", B_type="image")
            
            train_kernel(args, optimizer, scaler, net, moving_image, fixed_image, moving_label, fixed_label, moving_cano, fixed_cano, iteration)

            iteration += 1
            step_callback(unwrapped_net)

            if iteration % args.save_period == 0:
                torch.save(optimizer.state_dict(), footsteps.output_dir + f"checkpoints/optimizer_weights_e{epoch}_iter{iteration}")
                torch.save(unwrapped_net.regis_net.state_dict(), footsteps.output_dir + f"checkpoints/network_weights_e{epoch}_iter{iteration}")

            if iteration % args.eval_period == 0:
                unwrapped_net.eval()
                with torch.no_grad():
                    total_all_loss = 0.0
                    total_similarity_loss = 0.0
                    total_mono_similarity_loss = 0.0
                    total_inverse_consistency_loss = 0.0
                    total_can_cycle_consistency_loss = 0.0
                    total_transform_magnitude = 0.0
                    total_flips = 0.0
                    total_dice_score = 0.0
                    for src_pid, tgt_pid in tqdm(zip(args.test_src_pid, args.test_tgt_pid)):
                        logging.info(f"Test: {src_pid} -> {tgt_pid}")

                        patient_idx = {pid: idx for idx, pid in enumerate(val_dataset.data_dict['PatientID'])}
                        src_idx, tgt_idx = patient_idx[src_pid], patient_idx[tgt_pid]

                        src_img = minmax_norm(val_dataset.data_dict['T2'][src_idx].unsqueeze(0).unsqueeze(0).cuda())
                        tgt_img = minmax_norm(val_dataset.data_dict['b0'][tgt_idx].unsqueeze(0).unsqueeze(0).cuda())

                        src_seg = val_dataset.data_dict['T2_seg'][src_idx].unsqueeze(0).unsqueeze(0).float().cuda()
                        tgt_seg = val_dataset.data_dict['b0_seg'][tgt_idx].unsqueeze(0).unsqueeze(0).float().cuda()

                        test_loss = unwrapped_net(src_img, tgt_img, src_img, tgt_img, src_seg, tgt_seg, dice_logging=True)

                        logging.info(to_floats_generic(args, test_loss))

                        # loss
                        total_all_loss += test_loss.all_loss.item()
                        total_similarity_loss += test_loss.similarity_loss.item()
                        if args.training_type !='basic':
                            total_inverse_consistency_loss += test_loss.inverse_consistency_loss.item()
                            total_transform_magnitude += test_loss.transform_magnitude.item()
                            total_flips += test_loss.flips.item()
                        if args.training_type not in ['basic', 'basic+']:
                            total_can_cycle_consistency_loss += test_loss.canonical_consistency_loss.item()
                        if args.log_mono:
                            total_mono_similarity_loss += test_loss.mono_similarity_loss.item()
                        if args.training_type not in ['basic', 'basic+', 'unsupervised']:
                            total_dice_score += test_loss.Dice_score.item()
                        unwrapped_net.clean()

                total_all_loss = total_all_loss / len(args.test_src_pid)
                total_similarity_loss = total_similarity_loss / len(args.test_src_pid)
                total_inverse_consistency_loss = total_inverse_consistency_loss / len(args.test_src_pid)
                total_can_cycle_consistency_loss = total_can_cycle_consistency_loss / len(args.test_src_pid)
                if args.log_mono:
                    total_mono_similarity_loss = total_mono_similarity_loss / len(args.test_src_pid)
                total_transform_magnitude = total_transform_magnitude / len(args.test_src_pid)
                total_flips = total_flips / len(args.test_src_pid)
                total_dice_score = total_dice_score / len(args.test_src_pid)

                if args.training_type == 'basic':
                    total_avg_loss = BASICLoss(total_all_loss, total_similarity_loss)
                elif args.training_type == 'basic+':
                    total_avg_loss = ICONLoss(total_all_loss, total_similarity_loss, total_inverse_consistency_loss,
                                              total_transform_magnitude, total_flips)
                elif args.training_type == 'unsupervised':
                    if args.log_mono:
                        total_avg_loss = ICONLoss_can_mono_unsupervised(total_all_loss, total_similarity_loss,
                                                        total_mono_similarity_loss, total_inverse_consistency_loss,
                                                        total_can_cycle_consistency_loss, total_transform_magnitude,
                                                        total_flips)
                    else:
                        total_avg_loss = ICONLoss_can_unsupervised(total_all_loss, total_similarity_loss, total_inverse_consistency_loss,
                                                    total_can_cycle_consistency_loss, total_transform_magnitude,
                                                    total_flips)
                else:
                    if args.log_mono:
                        total_avg_loss = ICONLoss_can_mono(total_all_loss, total_similarity_loss,
                                                        total_mono_similarity_loss, total_inverse_consistency_loss,
                                                        total_can_cycle_consistency_loss, total_transform_magnitude,
                                                        total_flips, total_dice_score)
                    else:
                        total_avg_loss = ICONLoss_can(total_all_loss, total_similarity_loss, total_inverse_consistency_loss,
                                                    total_can_cycle_consistency_loss, total_transform_magnitude,
                                                    total_flips, total_dice_score)
                write_stats(args, total_avg_loss, iteration, prefix="val_avg/")
                unwrapped_net.train()
            ############################################################################################################

        train_end_time = time.time()
        train_time = train_end_time - train_start_time

        # if epoch % args.save_period == 0:
        #     torch.save(optimizer.state_dict(), footsteps.output_dir + f"checkpoints/optimizer_weights_e{epoch}_iter{iteration}")
        #     torch.save(unwrapped_net.regis_net.state_dict(), footsteps.output_dir + f"checkpoints/network_weights_e{epoch}_iter{iteration}")

        # Upload images to wandb
        val_start_time = time.time()
        if args.training_type not in ['basic', 'basic+', 'unsupervised']:
            visualization_moving, visualization_fixed, moving_label, fixed_label, moving_cano, fixed_cano = next(iter(val_loader))
            visualization_moving, visualization_fixed, moving_label, fixed_label = visualization_moving.cuda(), visualization_fixed.cuda(), moving_label.cuda(), fixed_label.cuda()
            moving_cano, fixed_cano = moving_cano.cuda(), fixed_cano.cuda()
        else:
            visualization_moving, visualization_fixed = next(iter(val_loader))
            visualization_moving, visualization_fixed = visualization_moving.cuda(), visualization_fixed.cuda()
            moving_label, fixed_label, moving_cano, fixed_cano = None, None, None, None

        unwrapped_net.eval()
        warped = []
        with torch.no_grad():
            eval_loss = unwrapped_net(visualization_moving, visualization_fixed, moving_cano, fixed_cano, moving_label, fixed_label, dice_logging=False)
            warped = unwrapped_net.warped_image_A.cpu()
            del eval_loss
            unwrapped_net.clean()
        unwrapped_net.train()

        def render(args, im, axis='axial'):
            if "ADNI" in args.dataset:
                if axis == 'axial':
                    im = im[:, :, :, im.shape[3] // 2]
                    im = torch.rot90(im, k=1, dims=(-2,-1))
                elif axis == 'sagittal':
                    im = im[:, :, im.shape[2] // 2]
                if torch.min(im) < 0:
                    im = im - torch.min(im)
                # if torch.max(im) > 1:
                im = im / torch.max(im)
                return im[:4, [0, 0, 0]].detach().cpu()

            elif args.dataset == "AutoPET":
                if axis == 'coronal':
                    im = im[:, :, :, im.shape[3] // 2]
                    im = torch.rot90(im, k=1, dims=(-2,-1))
                elif axis == 'axial':
                    im = im[:, :, :, :, im.shape[4] // 2]
                    im = torch.rot90(im, k=1, dims=(-2,-1))
                    im = torch.flip(im, dims=[-1])
                if torch.min(im) < 0:
                    im = im - torch.min(im)
                # if torch.max(im) > 1:
                im = im / torch.max(im)
                return im[:4, [0, 0, 0]].detach().cpu()

        if "ADNI" in args.dataset:
            wandb.log({
                "val_image/axial/moving_image": [wandb.Image(render(args, visualization_moving[:4], axis='axial'), mode="RGB", caption="Moving Image")],
                "val_image/axial/fixed_image": [wandb.Image(render(args, visualization_fixed[:4], axis='axial'), mode="RGB", caption="Fixed Image")],
                "val_image/axial/warped_moving_image": [wandb.Image(render(args, warped, axis='axial'), mode="RGB", caption="Warped Moving Image")],
                "val_image/axial/difference": [wandb.Image(render(args, torch.clip((warped[:4, :1] - visualization_fixed[:4, :1].cpu()) + 0.5, 0, 1), axis='axial'), mode="RGB", caption="Difference")]
            }, step=iteration)
            wandb.log({
                "val_image/sagittal/moving_image": [wandb.Image(render(args, visualization_moving[:4], axis='sagittal'), mode="RGB", caption="Moving Image")],
                "val_image/sagittal/fixed_image": [wandb.Image(render(args, visualization_fixed[:4], axis='sagittal'), mode="RGB", caption="Fixed Image")],
                "val_image/sagittal/warped_moving_image": [wandb.Image(render(args, warped, axis='sagittal'), mode="RGB", caption="Warped Moving Image")],
                "val_image/sagittal/difference": [wandb.Image(render(args, torch.clip((warped[:4, :1] - visualization_fixed[:4, :1].cpu()) + 0.5, 0, 1), axis='sagittal'), mode="RGB", caption="Difference")]
            }, step=iteration)

        elif args.dataset == "AutoPET":
            wandb.log({
                "val_image/coronal/moving_image": [wandb.Image(render(args, visualization_moving[:4], axis='coronal'), mode="RGB", caption="Moving Image")],
                "val_image/coronal/fixed_image": [wandb.Image(render(args, visualization_fixed[:4], axis='coronal'), mode="RGB", caption="Fixed Image")],
                "val_image/coronal/warped_moving_image": [wandb.Image(render(args, warped, axis='coronal'), mode="RGB", caption="Warped Moving Image")],
                "val_image/coronal/difference": [wandb.Image(render(args, torch.clip((warped[:4, :1] - visualization_fixed[:4, :1].cpu()) + 0.5, 0, 1), axis='coronal'), mode="RGB", caption="Difference")]
            }, step=iteration)
            wandb.log({
                "val_image/axial/moving_image": [wandb.Image(render(args, visualization_moving[:4], axis='axial'), mode="RGB", caption="Moving Image")],
                "val_image/axial/fixed_image": [wandb.Image(render(args, visualization_fixed[:4], axis='axial'), mode="RGB", caption="Fixed Image")],
                "val_image/axial/warped_moving_image": [wandb.Image(render(args, warped, axis='axial'), mode="RGB", caption="Warped Moving Image")],
                "val_image/axial/difference": [wandb.Image(render(args, torch.clip((warped[:4, :1] - visualization_fixed[:4, :1].cpu()) + 0.5, 0, 1), axis='axial'), mode="RGB", caption="Difference")]
            }, step=iteration)

        val_end_time = time.time()
        val_time = val_end_time - val_start_time

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        logging.info(f"Epoch {epoch} completed: Total Time = {epoch_time:.2f}s, Training Time = {train_time:.2f}s, Validation Time = {val_time:.2f}s")

    torch.save(optimizer.state_dict(), footsteps.output_dir + f"checkpoints/optimizer_weights_e{epoch}_iter{iteration}")
    torch.save(unwrapped_net.regis_net.state_dict(), footsteps.output_dir + f"checkpoints/network_weights_e{epoch}_iter{iteration}")
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    logging.info(f"Total training completed in {total_training_time:.2f}s")

def train_two_stage(args, train_loader, val_dataset, GPUS):
    torch.cuda.set_device(args.gpu_ids[0])
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    net = make_network(args, include_last_step=False, loss_fn=MINDSSC(),
                       use_label=False)

    # Continue train
    if args.resume_from != "":
        logging.info(f"Resume from: {args.resume_from}")
        net.regis_net.load_state_dict(torch.load(args.resume_from, map_location="cpu"))

    net_par = net.cuda() if GPUS == 1 else torch.nn.DataParallel(net, device_ids=args.gpu_ids, output_device=args.gpu_ids[0]).cuda()
    optimizer = torch.optim.Adam(net_par.parameters(), lr=args.lr)

    if args.resume_from != "":
        optimizer.load_state_dict(torch.load(args.resume_from.replace("network_weights_", "optimizer_weights_"), map_location="cpu"))

    logging.info("Start training...")
    net_par.train()
    train(args, net_par, optimizer, train_loader, val_dataset, unwrapped_net=net)

    torch.save(net.regis_net.state_dict(), footsteps.output_dir + "checkpoints/Step_1_final.pt",)

    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=False, default="first_test")
    parser.add_argument("--exp_dir", type=str, required=False, default="./results")
    parser.add_argument("--input_shape", type=int, required=False, default=128)
    parser.add_argument("--data_path", type=str, required=False, default="./datasets")
    parser.add_argument("--resume_from", type=str, required=False, default="")
    parser.add_argument("--epoch", type=int, required=False, default=100000)
    parser.add_argument("--batch", type=int, required=False, default=2)
    parser.add_argument("--save_period", type=int, required=False, default=5)
    parser.add_argument("--eval_period", type=int, required=False, default=30)
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids, 0,1,2,3 cpu=-1')
    parser.add_argument('--model', type=str, choices=['gradicon', 'transmorph', 'corrmlp'], required=False, default='gradicon')
    parser.add_argument('--training_type', type=str, choices=['basic', 'basic+', 'unsupervised', 'weakly-supervised', 'supervised'], required=False, default='basic')
    parser.add_argument('--dataset', type=str, required=False, help='pelvic_T2-b0', default = 'pelvic_T2-b0')
    parser.add_argument('--num_workers', type=int, required=False, default=0)
    parser.add_argument('--data_num', type=int, required=False, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lambda_inv', type=float, required=False, default=1.5)
    parser.add_argument('--lambda_can', type=float, required=False, default=1.5)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--log_mono', action='store_true', default=False)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--small', action='store_true', default=False)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.resume_from != "":
        args.exp_name = args.resume_from.split("/")[-4]

    if args.debug:
        args.exp_name = "debug_" + args.exp_name
    args.exp_dir = os.path.join(args.exp_dir, args.dataset, args.exp_name)
    os.makedirs(args.exp_dir, exist_ok=True)

    args.gpu_ids = [int(i) for i in args.gpu_ids.split(',')]
    GPUS = len(args.gpu_ids)

    setup_logging(args.exp_dir)
    logging.info(f"Start experiment: {args.exp_name}")
    logging.info(f"args:\n{args}")

    footsteps.initialize(run_name=args.exp_name, output_root=args.exp_dir)

    # WandB
    if args.resume_from != "":
        api = wandb.Api()
        runs = api.runs("MICCAI2025/M2M-Reg")

        for run in runs:
            if run.name == args.exp_name:
                wandb.init(project="M2M-Reg", name=args.exp_name, config=args, resume="must", id=run.id)
                break
    else:
        wandb.init(project="M2M-Reg", name=args.exp_name, config=args)

    train_dataset = dataset_multi.pelvic_T2_b0_Dataset(args=args, which_set='train')
    temp = train_dataset[0][0].shape
    args.input_shape = [1, 1, temp[1], temp[2], temp[3]]
    del temp

    temp_data_num = args.data_num
    if not args.debug:
        args.data_num = 100000000

    val_dataset = dataset_multi.pelvic_T2_b0_Dataset(args=args, which_set='val')
    # args.test_src_pid = {pid: idx for idx, pid in enumerate(val_dataset.data_dict['PatientID'])}
    # args.test_tgt_pid = {pid: idx for idx, pid in enumerate(val_dataset.data_dict['PatientID'])}
    args.test_src_pid = list(val_dataset.data_dict['PatientID'])
    args.test_tgt_pid = list(val_dataset.data_dict['PatientID'])

    args.data_num = temp_data_num

    train_loader = DataLoader(train_dataset, batch_size=args.batch * GPUS, shuffle = True,
                              num_workers=args.num_workers, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True,
                            num_workers=args.num_workers, drop_last=True)
    logging.info("Finish data loading...")

    os.makedirs(footsteps.output_dir + "checkpoints", exist_ok=True)

    logging.info("Start training...")
    train_two_stage(args, train_loader, val_dataset, GPUS)
