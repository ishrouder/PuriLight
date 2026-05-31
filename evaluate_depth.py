from __future__ import absolute_import, division, print_function
import csv
import os
import re
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import cv2

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from layers import disp_to_depth
from utils import readlines
from options import PuriLightXOptions
import datasets
import networks
import time
from thop import clever_format
from thop import profile


cv2.setNumThreads(0)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

torch.backends.cudnn.benchmark = True


def profile_once(encoder, decoder, x):
    x_e = x[0, :, :, :].unsqueeze(0)
    x_d = encoder(x_e)
    flops_e, params_e = profile(encoder, inputs=(x_e, ), verbose=False)
    flops_d, params_d = profile(decoder, inputs=(x_d, ), verbose=False)

    flops, params = clever_format([flops_e + flops_d, params_e + params_d], "%.3f")
    flops_e, params_e = clever_format([flops_e, params_e], "%.3f")
    flops_d, params_d = clever_format([flops_d, params_d], "%.3f")

    return flops, params, flops_e, params_e, flops_d, params_d


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    device = torch.device("cpu" if opt.no_cuda else "cuda")

    # Only iterate over epoch numbers if the folder name ends with a plain number
    # (e.g., weights_19). Skip iteration for named folders like kitti_lf_640x192.
    base_name = os.path.basename(opt.load_weights_folder.rstrip('/\\'))
    match = re.search(r'^.*?(\d+)$', base_name)
    if match and not re.search(r'\d+x\d+', base_name):
        number = int(match.group(1))
    else:
        match = None
        number = 0

    csv_rows = []
    csv_path = os.path.join(os.path.dirname(opt.load_weights_folder), "eval_results.csv")

    while number >= 0:

        if match:
            opt.load_weights_folder = re.sub(r'(\d+)(?!.*\d)', str(number), opt.load_weights_folder)

        if not os.path.isdir(opt.load_weights_folder):
            print("-> Skipping {}, folder not found".format(opt.load_weights_folder))
            number -= 1
            continue

        if opt.ext_disp_to_eval is None:

            print("-> Loading weights from {}".format(opt.load_weights_folder))

            filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
            encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
            decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

            encoder_dict = torch.load(encoder_path, map_location=device)
            decoder_dict = torch.load(decoder_path, map_location=device)

            # Allow command-line height/width to override checkpoint values
            eval_height = opt.height if opt.height != 192 else encoder_dict.get('height', opt.height)
            eval_width = opt.width if opt.width != 640 else encoder_dict.get('width', opt.width)

            if opt.eval_split == 'cityscapes':
                dataset = datasets.CityscapesEvalDataset(opt.data_path, filenames,
                                                         eval_height, eval_width,
                                                         [0], 4,
                                                         is_train=False)
            else:
                img_ext = '.png' if opt.png else '.jpg'
                dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                                   eval_height, eval_width,
                                                   [0], 4, is_train=False, img_ext=img_ext)
            dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                    pin_memory=True, drop_last=False)

            encoder = networks.PuriLightX(model=opt.model,
                                        height=eval_height,
                                        width=eval_width)
            depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
            model_dict = encoder.state_dict()
            depth_model_dict = depth_decoder.state_dict()
            encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
            depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

            encoder.to(device)
            encoder.eval()
            depth_decoder.to(device)
            depth_decoder.eval()

            pred_disps = []
            if opt.eval_doj:
                object_masks = []

            print("-> Computing predictions with size {}x{}".format(
                eval_width, eval_height))

            with torch.no_grad():
                for data in dataloader:
                    input_color = data[("color", 0, 0)].to(device)

                    if opt.post_process:
                        input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                    if opt.eval_split == 'cityscapes':
                        if opt.eval_doj:
                            for bi in range(input_color.size(0)):
                                object_masks.append(data["doj_mask"][bi])

                    flops, params, flops_e, params_e, flops_d, params_d = profile_once(encoder, depth_decoder, input_color)
                    output = depth_decoder(encoder(input_color))

                    pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                    pred_disp = pred_disp.cpu()[:, 0].numpy()

                    if opt.post_process:
                        N = pred_disp.shape[0] // 2
                        pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                    pred_disps.append(pred_disp)

            pred_disps = np.concatenate(pred_disps)

        else:
            print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
            pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.save_pred_disps:
            output_path = os.path.join(
                opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
            print("-> Saving predicted disparities to ", output_path)
            np.save(output_path, pred_disps)

        if opt.no_eval:
            print("-> Evaluation disabled. Done.")
            quit()

        if opt.eval_split == 'cityscapes':
            print('loading cityscapes gt depths individually due to their combined size!')
            gt_depths = os.path.join(splits_dir, opt.eval_split, "gt_depths")
        else:
            gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
            gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

        print("-> Evaluating")
        print("   Mono evaluation - using median scaling")

        errors = []
        ratios = []
        if opt.eval_doj:
            object_errors = []
            object_pixels = []
            static_errors = []
            static_pixels = []

        for i in range(pred_disps.shape[0]):
            if opt.eval_split == 'cityscapes':
                gt_depth = np.load(os.path.join(gt_depths, str(i).zfill(3) + '_depth.npy'))
                gt_height, gt_width = gt_depth.shape[:2]
                gt_height = int(round(gt_height * 0.75))
                gt_depth = gt_depth[:gt_height]

            else:
                gt_depth = gt_depths[i]
                gt_height, gt_width = gt_depth.shape[:2]

            pred_disp = pred_disps[i]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            if opt.eval_split == 'cityscapes':
                gt_depth = gt_depth[256:, 192:1856]
                pred_depth = pred_depth[256:, 192:1856]

            if opt.eval_split == "eigen":
                mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                 0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

            elif opt.eval_split == 'cityscapes':
                mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
                if opt.eval_doj:
                    object_mask = F.interpolate(object_masks[i].unsqueeze(0), [gt_height, gt_width])
                    object_mask = object_mask[0][0][256:, 192:1856].cpu().numpy()

            else:
                mask = gt_depth > 0

            pred_depth *= opt.pred_depth_scale_factor

            if not opt.disable_median_scaling:
                ratio = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
                ratios.append(ratio)
                pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            errors.append(compute_errors(gt_depth[mask], pred_depth[mask]))

            if opt.eval_doj:
                doj_mask = np.logical_and(mask, object_mask)
                stc_mask = np.logical_and(mask, 1 - doj_mask)
                static_errors.append(compute_errors(gt_depth[stc_mask], pred_depth[stc_mask]))
                static_pixels.append(stc_mask.sum().item())
                if doj_mask.sum() != 0:
                    object_errors.append(compute_errors(gt_depth[doj_mask], pred_depth[doj_mask]))
                    object_pixels.append(doj_mask.sum().item())

        if not opt.disable_median_scaling:
            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

        mean_errors = np.array(errors).mean(0)

        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n  " + ("flops: {0}, params: {1}, flops_e: {2}, params_e:{3}, flops_d:{4}, params_d:{5}").format(flops, params, flops_e, params_e, flops_d, params_d))

        csv_rows.append([number] + mean_errors.tolist())

        if opt.eval_doj:
            static_pixels = np.array(static_pixels, dtype=np.float64)
            static_errors = np.array(static_errors, dtype=np.float64)
            static_sum = static_pixels.sum()
            static_pixels_s = static_pixels / static_sum
            static_pixels_s = static_pixels_s[:, np.newaxis]
            mean_stc_errors = (static_errors * static_pixels_s).sum(0)

            object_pixels = np.array(object_pixels, dtype=np.float64)
            object_errors = np.array(object_errors, dtype=np.float64)
            object_sum = object_pixels.sum()
            object_pixels_s = object_pixels / object_sum
            object_pixels_s = object_pixels_s[:, np.newaxis]
            mean_obj_errors = (object_errors * object_pixels_s).sum(0)

            print("\nMetrics on dynamic object region\n  " + ("{:>8} | " * 7).format("abs_rel",
                                                                                     "sq_rel", "rmse", "rmse_log", "a1",
                                                                                     "a2", "a3"))
            print(("&{: 8.3f}  " * 7).format(*mean_obj_errors.tolist()) + "\\\\")

            print("\nMetrics on static scene region\n  " + ("{:>8} | " * 7).format("abs_rel",
                                                                                     "sq_rel", "rmse", "rmse_log", "a1",
                                                                                     "a2", "a3"))
            print(("&{: 8.3f}  " * 7).format(*mean_stc_errors.tolist()) + "\\\\")

        if not match:
            break
        number -= 1

    if csv_rows:
        csv_rows.sort(key=lambda x: x[0])
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"])
            writer.writerows(csv_rows)
        print("-> Eval results saved to {}".format(csv_path))

    print("\n-> Done!")


if __name__ == "__main__":
    options = PuriLightXOptions()
    evaluate(options.parse())