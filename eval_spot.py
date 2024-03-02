import copy
import os.path
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision.utils import save_image
from spot import SPOT
from datasets import PascalVOC, COCO2017, MOVi
from ocl_metrics import UnsupervisedMaskIoUMetric, ARIMetric
from utils_spot import inv_normalize, visualize, bool_flag
import models_vit

parser = argparse.ArgumentParser()

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--val_image_size', type=int, default=224)
parser.add_argument('--val_mask_size', type=int, default=320)
parser.add_argument('--eval_batch_size', type=int, default=32)
parser.add_argument('--viz_resolution_factor', type=float, default=0.5)

parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
parser.add_argument('--log_path', default='results')
parser.add_argument('--dataset', default='coco', help='coco or voc')
parser.add_argument('--data_path',  type=str, help='dataset path')

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
parser.add_argument('--finetune_blocks_after',  type=int, default=100, help='just use a large number')
parser.add_argument('--encoder_final_norm',  type=bool_flag, default=False)

parser.add_argument('--truncate',  type=str, default='bi-level', help='bi-level or fixed-point or none')
parser.add_argument('--init_method', default='embedding', help='embedding or shared_gaussian')

parser.add_argument('--use_second_encoder',  type= bool_flag, default = True, help='different encoder for input and target of decoder')

parser.add_argument('--train_permutations',  type=str, default='random', help='it is just for the initialization')
parser.add_argument('--eval_permutations',  type=str, default='standard', help='standard, random, or all')

args = parser.parse_args()

torch.manual_seed(args.seed)

arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
arg_str = '__'.join(arg_str_list)
log_dir = os.path.join(args.log_path, os.path.basename(os.path.dirname(args.checkpoint_path)))
os.makedirs(log_dir, exist_ok=True)

if args.dataset == 'voc':
    val_dataset = PascalVOC(root=args.data_path, split='val', image_size=args.val_image_size, mask_size = args.val_mask_size)
elif args.dataset == 'coco':
    val_dataset = COCO2017(root=args.data_path, split='val', image_size=args.val_image_size, mask_size = args.val_mask_size)
elif args.dataset == 'movi':
    val_dataset = MOVi(root=os.path.join(args.data_path, 'validation'), split='validation', image_size=args.val_image_size, mask_size = args.val_mask_size)

args.max_tokens = int((args.val_image_size/16)**2)

val_sampler = None

loader_kwargs = {
    'num_workers': args.num_workers,
    'pin_memory': True,
}

val_loader = DataLoader(val_dataset, sampler=val_sampler, shuffle=False, drop_last = False, batch_size=args.eval_batch_size, **loader_kwargs)

val_epoch_size = len(val_loader)

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
 
else:
    raise
        
encoder = encoder.eval()

if args.use_second_encoder:
    encoder_second = copy.deepcopy(encoder).eval()
else:
    encoder_second = None

if args.num_cross_heads is None:
    args.num_cross_heads = args.num_heads

model = SPOT(encoder, args, encoder_second)

checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
checkpoint['model'] = {k.replace("tf_dec.", "dec."): v for k, v in checkpoint['model'].items()} # compatibility with older runs
model.load_state_dict(checkpoint['model'], strict = True)

model = model.cuda()

MBO_c_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).cuda()
MBO_i_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).cuda()
miou_metric = UnsupervisedMaskIoUMetric(matching="hungarian", ignore_background = True, ignore_overlaps = True).cuda()
ari_metric = ARIMetric(foreground = True, ignore_overlaps = True).cuda()

MBO_c_slot_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).cuda()
MBO_i_slot_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).cuda()
miou_slot_metric = UnsupervisedMaskIoUMetric(matching="hungarian", ignore_background = True, ignore_overlaps = True).cuda()
ari_slot_metric = ARIMetric(foreground = True, ignore_overlaps = True).cuda()

with torch.no_grad():
    model.eval()

    val_mse = 0.
    counter = 0

    for batch, (image, true_mask_i, true_mask_c, mask_ignore) in enumerate(tqdm(val_loader)):
        image = image.cuda()
        true_mask_i = true_mask_i.cuda()
        true_mask_c = true_mask_c.cuda()
        mask_ignore = mask_ignore.cuda() 
        
        batch_size = image.shape[0]
        counter += batch_size

        mse, default_slots_attns, dec_slots_attns, _, _, _ = model(image)

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
    ari_slot = 100 * ari_slot_metric.compute()
    mbo_c = 100 * MBO_c_metric.compute()
    mbo_i = 100 * MBO_i_metric.compute()
    miou = 100 * miou_metric.compute()
    mbo_c_slot = 100 * MBO_c_slot_metric.compute()
    mbo_i_slot = 100 * MBO_i_slot_metric.compute()
    miou_slot = 100 * miou_slot_metric.compute()
    val_loss = val_mse

    df_results = pd.DataFrame([[mbo_i.item(), mbo_c.item(), ari.item(),  val_mse, mbo_i_slot.item(), mbo_c_slot.item(), ari_slot.item(), miou.item(), miou_slot.item()]], 
                 columns=['mBO_i', 'mBO_c', 'FG-ARI',  'MSE', 'mBO_i_slots', 'mBO_c_slots', 'FG-ARI_slots', 'miou', 'miou_slots'])
    
    print(args.checkpoint_path)
    print(df_results)
    
    # For plotting
    image = inv_normalize(image)
    image = F.interpolate(image, size=args.val_mask_size, mode='bilinear')
    rgb_default_attns = image.unsqueeze(1) * default_attns + 1. - default_attns
    rgb_dec_attns = image.unsqueeze(1) * dec_attns + 1. - dec_attns
    
    vis_recon = visualize(image, true_mask_c, pred_dec_mask, rgb_dec_attns, pred_default_mask, rgb_default_attns, N=32)
    grid = vutils.make_grid(vis_recon, nrow=2*args.num_slots + 4, pad_value=0.2)[:, 2:-2, 2:-2]
    grid = F.interpolate(grid.unsqueeze(1), scale_factor=args.viz_resolution_factor, mode='bilinear').squeeze() # Lower resolution
    save_image(grid, os.path.join(log_dir,'output.png'))
