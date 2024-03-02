import math
import os.path
import argparse
from tqdm import tqdm
from datetime import datetime
import copy
import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from torch.nn import CrossEntropyLoss
from spot import SPOT
from datasets import PascalVOC, COCO2017, MOVi
from ocl_metrics import UnsupervisedMaskIoUMetric, ARIMetric
from utils_spot import inv_normalize, cosine_scheduler, visualize, att_matching, bool_flag, load_pretrained_encoder
import models_vit
IGNORE_INDEX = -100

def get_args_parser():
    parser = argparse.ArgumentParser('SPOT (2)', add_help=False)
    
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--clip', type=float, default=0.3)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--val_image_size', type=int, default=224)
    parser.add_argument('--val_mask_size', type=int, default=320)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--eval_viz_percent', type=float, default=0.2)
    
    parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar', help='checkpoint to continue the training, loaded only if exists')
    parser.add_argument('--log_path', default='logs')
    parser.add_argument('--dataset', default='coco', help='coco or voc')
    parser.add_argument('--data_path',  type=str, help='dataset path')
    parser.add_argument('--predefined_movi_json_paths', default = None,  type=str, help='For MOVi datasets, use the same subsampled images. Typically for the 2nd stage of Spot training to retain the same images')
    
    parser.add_argument('--lr_main', type=float, default=4e-4)
    parser.add_argument('--lr_min', type=float, default=4e-7)
    parser.add_argument('--lr_warmup_steps', type=int, default=10000)
    
    parser.add_argument('--num_dec_blocks', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    parser.add_argument('--num_iterations', type=int, default=3)
    parser.add_argument('--num_slots', type=int, default=7)
    parser.add_argument('--slot_size', type=int, default=256)
    parser.add_argument('--mlp_hidden_size', type=int, default=1024)
    parser.add_argument('--img_channels', type=int, default=3)
    parser.add_argument('--pos_channels', type=int, default=4)
    parser.add_argument('--num_cross_heads', type=int, default=None)
    
    parser.add_argument('--dec_type',  type=str, default='transformer', help='type of decoder transformer or mlp')
    parser.add_argument('--cappa', type=float, default=-1)
    parser.add_argument('--mlp_dec_hidden',  type=int, default=2048, help='Dimension of decoder mlp hidden layers')
    parser.add_argument('--use_slot_proj',  type=bool_flag, default=True, help='Use an extra projection before MLP decoder')

    parser.add_argument('--which_encoder',  type=str, default='dino_vitb16', help='dino_vitb16, dino_vits8, dinov2_vitb14_reg, dinov2_vits14_reg, dinov2_vitb14, dinov2_vits14, mae_vitb16')
    parser.add_argument('--finetune_blocks_after',  type=int, default=8, help='finetune the blocks from this and after (counting from 0), for vit-b values greater than 12 means keep everything frozen')
    parser.add_argument('--encoder_final_norm',  type=bool_flag, default=False)
    parser.add_argument('--pretrained_encoder_weights', type=str, default=None)
    
    parser.add_argument('--truncate',  type=str, default='bi-level', help='bi-level or fixed-point or none')
    parser.add_argument('--init_method', default='embedding', help='embedding or shared_gaussian')
    
    parser.add_argument('--train_permutations',  type=str, default='random', help='which permutation')
    parser.add_argument('--eval_permutations',  type=str, default='standard', help='which permutation')
    
    parser.add_argument('--ce_weight', type=float, default=5e-3, help='weight of the cross-entropy distilation loss')
    parser.add_argument('--final_ce_weight', type=float, default=None, help='final weight of the cross-entropy distilation loss')
    
    parser.add_argument('--teacher_checkpoint_path', help='teacher checkpoint')
    parser.add_argument('--teacher_truncate',  type=str, default = 'none')
    parser.add_argument('--teacher_init_method',  type=str, default = 'shared_gaussian')
    parser.add_argument('--teacher_train_permutations',  type=str, default='random', help='which permutation')
    parser.add_argument('--teacher_eval_permutations',  type=str, default='random', help='which permutation')
    
    return parser

def train(args):
    torch.manual_seed(args.seed)
    
    arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
    arg_str = '__'.join(arg_str_list)
    log_dir = os.path.join(args.log_path, datetime.today().isoformat())
    writer = SummaryWriter(log_dir)
    writer.add_text('hparams', arg_str)
    
    if args.dataset == 'voc':
        train_dataset = PascalVOC(root=args.data_path, split='trainaug', image_size=args.image_size, mask_size = args.image_size)
        val_dataset = PascalVOC(root=args.data_path, split='val', image_size=args.val_image_size, mask_size = args.val_mask_size)
    elif args.dataset == 'coco':
        train_dataset = COCO2017(root=args.data_path, split='train', image_size=args.image_size, mask_size = args.image_size)
        val_dataset = COCO2017(root=args.data_path, split='val', image_size=args.val_image_size, mask_size = args.val_mask_size)
    elif args.dataset == 'movi':
        train_dataset = MOVi(root=os.path.join(args.data_path, 'train'), split='train', image_size=args.image_size, mask_size = args.image_size, frames_per_clip=9, predefined_json_paths = args.predefined_movi_json_paths)
        val_dataset = MOVi(root=os.path.join(args.data_path, 'validation'), split='validation', image_size=args.val_image_size, mask_size = args.val_mask_size)

    train_sampler = None
    val_sampler = None
    
    loader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': True,
    }
    
    train_loader = DataLoader(train_dataset, sampler=train_sampler, shuffle=True, drop_last = True, batch_size=args.batch_size, **loader_kwargs)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, shuffle=False, drop_last = False, batch_size=args.eval_batch_size, **loader_kwargs)
    
    train_epoch_size = len(train_loader)
    val_epoch_size = len(val_loader)
    
    log_interval = train_epoch_size // 5
    
    if args.which_encoder == 'dino_vitb16':
        args.max_tokens = int((args.val_image_size/16)**2)
        encoder = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    elif args.which_encoder == 'dino_vits8':
        args.max_tokens = int((args.val_image_size/8)**2)
        encoder = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    elif args.which_encoder == 'dino_vitb8':
        args.max_tokens = int((args.val_image_size/8)**2)
        encoder = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
    elif args.which_encoder == 'dinov2_vitb14':
        args.max_tokens = int((args.val_image_size/14)**2)
        encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    elif args.which_encoder == 'dinov2_vits14':
        args.max_tokens = int((args.val_image_size/14)**2)
        encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    elif args.which_encoder == 'dinov2_vitb14_reg':
        args.max_tokens = int((args.val_image_size/14)**2)
        encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    elif args.which_encoder == 'dinov2_vits14_reg':
        args.max_tokens = int((args.val_image_size/14)**2)
        encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    elif args.which_encoder == 'mae_vitb16':
        args.max_tokens = int((args.val_image_size/16)**2)
        encoder = models_vit.__dict__["vit_base_patch16"](num_classes=0, global_pool=False, drop_path_rate=0)
        assert args.pretrained_encoder_weights is not None
        load_pretrained_encoder(encoder, args.pretrained_encoder_weights, prefix=None)      
    else:
        raise
    
    encoder_new = copy.deepcopy(encoder).train()
    encoder = encoder.eval()
    
    if args.num_cross_heads is None:
        args.num_cross_heads = args.num_heads
    
    student_model = SPOT(encoder_new, args, encoder)
    
    args_teacher = copy.deepcopy(args)
    args_teacher.truncate = args.teacher_truncate
    args_teacher.init_method = args.teacher_init_method
    args_teacher.train_permutations = args.teacher_train_permutations
    args_teacher.eval_permutations = args.teacher_eval_permutations
    args_teacher.finetune_blocks_after = 100
    
    teacher_model = SPOT(encoder, args_teacher)

    checkpoint = torch.load(args.teacher_checkpoint_path, map_location='cpu')
    checkpoint['model'] = {k.replace("tf_dec.", "dec."): v for k, v in checkpoint['model'].items()} # compatibility with older runs
    teacher_model.load_state_dict(checkpoint['model'], strict=True)
    msg = teacher_model.load_state_dict(checkpoint['model'], strict=True)
    for param in teacher_model.parameters():
        param.requires_grad = False  # not update by gradient
    print(msg)

    if os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        best_val_ari = checkpoint['best_val_ari']
        best_val_ari_slot = checkpoint['best_val_ari_slot']
        best_mbo_c = checkpoint['best_mbo_c']
        best_mbo_i = checkpoint['best_mbo_i']
        best_miou = checkpoint['best_miou']
        best_mbo_c_slot = checkpoint['best_mbo_c_slot']
        best_mbo_i_slot = checkpoint['best_mbo_i_slot']
        best_miou_slot = checkpoint['best_miou_slot']
        best_epoch = checkpoint['best_epoch']
        student_model.load_state_dict(checkpoint['model'], strict=True)
        msg = student_model.load_state_dict(checkpoint['model'], strict=True)
        print(msg)
    else:
        print('No checkpoint_path found')
        checkpoint = None
        start_epoch = 0
        best_val_loss = math.inf
        best_epoch = 0
        best_val_ari = 0
        best_val_ari_slot = 0
        best_mbo_c = 0
        best_mbo_i = 0
        best_miou= 0 
        best_mbo_c_slot = 0
        best_mbo_i_slot = 0
        best_miou_slot= 0
    
    teacher_model = teacher_model.cuda()
    student_model = student_model.cuda()
    
    lr_schedule = cosine_scheduler( base_value = args.lr_main,
                                    final_value = args.lr_min,
                                    epochs = args.epochs, 
                                    niter_per_ep = len(train_loader),
                                    warmup_epochs=int(args.lr_warmup_steps/(len(train_dataset)/args.batch_size)),
                                    start_warmup_value=0)

    if args.final_ce_weight == None:
        args.final_ce_weight = args.ce_weight

    ce_weight_schedule = cosine_scheduler( base_value = args.ce_weight,
                                final_value = args.final_ce_weight,
                                epochs = args.epochs, 
                                niter_per_ep = len(train_loader),
                                warmup_epochs=0,
                                start_warmup_value=0)
    
    optimizer = Adam([
        {'params': (param for name, param in student_model.named_parameters() if param.requires_grad), 'lr': args.lr_main},
    ])
    
    criterion = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    
    MBO_c_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).cuda()
    MBO_i_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).cuda()
    miou_metric = UnsupervisedMaskIoUMetric(matching="hungarian", ignore_background = True, ignore_overlaps = True).cuda()
    ari_metric = ARIMetric(foreground = True, ignore_overlaps = True).cuda()
    
    MBO_c_slot_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).cuda()
    MBO_i_slot_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).cuda()
    miou_slot_metric = UnsupervisedMaskIoUMetric(matching="hungarian", ignore_background = True, ignore_overlaps = True).cuda()
    ari_slot_metric = ARIMetric(foreground = True, ignore_overlaps = True).cuda()
    
    visualize_per_epoch = int(args.epochs*args.eval_viz_percent)
    
    teacher_model.eval()
    for epoch in range(start_epoch, args.epochs):
    
        student_model.train()
    
        for batch, image in enumerate(train_loader):
            
            image = image.cuda()

            global_step = epoch * train_epoch_size + batch
    
            optimizer.param_groups[0]['lr'] = lr_schedule[global_step]
            lr_value = optimizer.param_groups[0]['lr']
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                _, _, dec_slots_attns, _, _, _ = teacher_model(image)
                dec_masks = dec_slots_attns.argmax(1)
                dec_masks_onehot = torch.nn.functional.one_hot(dec_masks, num_classes=args.num_slots).permute(0,3,1,2)
                B, H, W = dec_masks.size()
            
            mse, slots_attns, _, _, _, logits = student_model(image)
            
            logits = logits.transpose(-1, -2).reshape(B, args.num_slots, H, W)
            
            attn_onehot = torch.nn.functional.one_hot(slots_attns.argmax(1), num_classes=args.num_slots).permute(0,3,1,2)
            permutation_indices, _ = att_matching(attn_onehot, dec_masks_onehot)
        
            logits = torch.stack([x[permutation_indices[n]] for n, x in enumerate(logits)], dim=0)
        
            ce_loss = criterion(logits, dec_masks)

            ce_weight = ce_weight_schedule[global_step]

            total_loss = mse + ce_weight*ce_loss
            total_loss.backward()
            clip_grad_norm_(student_model.parameters(), args.clip, 'inf')
            optimizer.step()
            
            with torch.no_grad():
                if batch % log_interval == 0:
                    print('Train Epoch: {:3} [{:5}/{:5}] \t lr = {:5} \t MSE: {:F} \t CE: {:F}'.format(
                          epoch+1, batch, train_epoch_size, lr_value, mse.item(), ce_loss.item()))
    
                    writer.add_scalar('TRAIN/mse', mse.item(), global_step)
                    writer.add_scalar('TRAIN/ce', ce_loss.item(), global_step)
                    writer.add_scalar('TRAIN/lr_main', lr_value, global_step)

        with torch.no_grad():
            student_model.eval()

            val_mse = 0.
            counter = 0
    
            for batch, (image, true_mask_i, true_mask_c, mask_ignore) in enumerate(tqdm(val_loader)):
                image = image.cuda()
                true_mask_i = true_mask_i.cuda()
                true_mask_c = true_mask_c.cuda()
                mask_ignore = mask_ignore.cuda() 
                
                batch_size = image.shape[0]
                counter += batch_size
    
                mse, default_slots_attns, dec_slots_attns, _, _, _ = student_model(image)
    
                # DINOSAUR uses as attention masks the attenton maps of the decoder
                # over the slots, which bilinearly resizes to match the image resolution
                # dec_slots_attns shape: [B, num_slots, H_enc, W_enc]
                default_attns = F.interpolate(default_slots_attns, size=args.val_mask_size, mode='bilinear')
                dec_attns = F.interpolate(dec_slots_attns, size=args.val_mask_size, mode='bilinear')
                # dec_attns shape [B, num_slots, H, W]
                default_attns = default_attns.unsqueeze(2)
                dec_attns = dec_attns.unsqueeze(2) # shape [B, num_slots, 1, H, W]
    
                pred_default_mask = default_attns.argmax(1).squeeze(1)
                pred_dec_mask = dec_attns.argmax(1).squeeze(1)
    
                val_mse += mse.item()
                
                # Compute ARI, MBO_i and MBO_c, miou scores for both slot attention and decoder
                true_mask_i_reshaped = torch.nn.functional.one_hot(true_mask_i).to(torch.float32).permute(0,3,1,2).cuda()
                true_mask_c_reshaped = torch.nn.functional.one_hot(true_mask_c).to(torch.float32).permute(0,3,1,2).cuda()
                pred_dec_mask_reshaped = torch.nn.functional.one_hot(pred_dec_mask).to(torch.float32).permute(0,3,1,2).cuda()
                pred_default_mask_reshaped = torch.nn.functional.one_hot(pred_default_mask).to(torch.float32).permute(0,3,1,2).cuda()
                
                MBO_i_metric.update(pred_dec_mask_reshaped, true_mask_i_reshaped, mask_ignore)
                MBO_c_metric.update(pred_dec_mask_reshaped, true_mask_c_reshaped, mask_ignore)
                miou_metric.update(pred_dec_mask_reshaped, true_mask_i_reshaped, mask_ignore)
                ari_metric.update(pred_dec_mask_reshaped, true_mask_i_reshaped, mask_ignore)
            
                MBO_i_slot_metric.update(pred_default_mask_reshaped, true_mask_i_reshaped, mask_ignore)
                MBO_c_slot_metric.update(pred_default_mask_reshaped, true_mask_c_reshaped, mask_ignore)
                miou_slot_metric.update(pred_default_mask_reshaped, true_mask_i_reshaped, mask_ignore)
                ari_slot_metric.update(pred_default_mask_reshaped, true_mask_i_reshaped, mask_ignore)
    
            val_mse /= (val_epoch_size)
            ari = 100 * ari_metric.compute()
            mbo_c = 100 * MBO_c_metric.compute()
            mbo_i = 100 * MBO_i_metric.compute()
            miou = 100 * miou_metric.compute()
            ari_slot = 100 * ari_slot_metric.compute()
            mbo_c_slot = 100 * MBO_c_slot_metric.compute()
            mbo_i_slot = 100 * MBO_i_slot_metric.compute()
            miou_slot = 100 * miou_slot_metric.compute()
            val_loss = val_mse
            writer.add_scalar('VAL/mse', val_mse, epoch+1)
            writer.add_scalar('VAL/ari (slots)', ari_slot, epoch+1)
            writer.add_scalar('VAL/ari (decoder)', ari, epoch+1)
            writer.add_scalar('VAL/mbo_c', mbo_c, epoch+1)
            writer.add_scalar('VAL/mbo_i', mbo_i, epoch+1)
            writer.add_scalar('VAL/miou', miou, epoch+1)
            writer.add_scalar('VAL/mbo_c (slots)', mbo_c_slot, epoch+1)
            writer.add_scalar('VAL/mbo_i (slots)', mbo_i_slot, epoch+1)
            writer.add_scalar('VAL/miou (slots)', miou_slot, epoch+1)
            
            print(args.log_path)
            print('====> Epoch: {:3} \t Loss = {:F} \t MSE = {:F} \t ARI = {:F} \t ARI_slots = {:F} \t mBO_c = {:F} \t mBO_i = {:F} \t miou = {:F} \t mBO_c_slots = {:F} \t mBO_i_slots = {:F} \t miou_slots = {:F}'.format(
                epoch+1, val_loss, val_mse, ari, ari_slot, mbo_c, mbo_i, miou, mbo_c_slot, mbo_i_slot, miou_slot))
            
            ari_metric.reset()
            MBO_c_metric.reset()
            MBO_i_metric.reset()
            miou_metric.reset()
            MBO_c_slot_metric.reset()
            MBO_i_slot_metric.reset()
            ari_slot_metric.reset()
            miou_slot_metric.reset()
            
            if (val_loss < best_val_loss) or (best_val_ari > ari) or (best_mbo_c > mbo_c):
                best_val_loss = val_loss
                best_val_ari = ari
                best_val_ari_slot = ari_slot
                best_mbo_c = mbo_c
                best_mbo_i = mbo_i
                best_miou = miou
                best_mbo_c_slot = mbo_c_slot
                best_mbo_i_slot = mbo_i_slot
                best_miou_slot = miou_slot
                best_epoch = epoch + 1
    
                torch.save(student_model.state_dict(), os.path.join(log_dir, 'best_model.pt'))
                
            if epoch%visualize_per_epoch==0 or epoch==args.epochs-1:
                image = inv_normalize(image)
                image = F.interpolate(image, size=args.val_mask_size, mode='bilinear')
                rgb_default_attns = image.unsqueeze(1) * default_attns + 1. - default_attns
                rgb_dec_attns = image.unsqueeze(1) * dec_attns + 1. - dec_attns
    
                vis_recon = visualize(image, true_mask_c, pred_dec_mask, rgb_dec_attns, pred_default_mask, rgb_default_attns, N=32)
                grid = vutils.make_grid(vis_recon, nrow=2*args.num_slots + 4, pad_value=0.2)[:, 2:-2, 2:-2]
                grid = F.interpolate(grid.unsqueeze(1), scale_factor=0.15, mode='bilinear').squeeze() # Lower resolution
                writer.add_image('VAL_recon/epoch={:03}'.format(epoch + 1), grid)
    
            writer.add_scalar('VAL/best_loss', best_val_loss, epoch+1)
    
            checkpoint = {
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'best_val_ari': best_val_ari,
                'best_val_ari_slot': best_val_ari_slot,
                'best_mbo_c':best_mbo_c,
                'best_mbo_i':best_mbo_i,
                'best_miou':best_miou,
                'best_mbo_c_slot':best_mbo_c_slot,
                'best_mbo_i_slot':best_mbo_i_slot,
                'best_miou_slot':best_miou_slot,
                'best_epoch': best_epoch,
                'model': student_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
    
            torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pt.tar'))
    
            print('====> Best Loss = {:F} @ Epoch {}'.format(best_val_loss, best_epoch))
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SPOT (2)', parents=[get_args_parser()])
    args = parser.parse_args()
    train(args)