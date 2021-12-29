import argparse
import os
from threading import Condition

import cv2
import numpy as np
from metrics import psnr, ssim
from inference import init_logger, BasicVSR, get_image_list
import torch
from misc import tensor2img

def parse_args():
    parser = argparse.ArgumentParser(description='BasicVSR Inference')
    parser.add_argument('lrdir', help='Low resolution dir')
    parser.add_argument('gtdir', help='Ground truth dir')
    parser.add_argument(
        '--spynet-model',
        default='models/spynet.compilation/compilation.bmodel',
        help='SPyNet model path')
    parser.add_argument(
        '--backward-residual-model',
        default='models/backward_residual.compilation/compilation.bmodel',
        help='Backward residual model path')
    parser.add_argument(
        '--forward-residual-model',
        default='models/forward_residual.compilation/compilation.bmodel',
        help='Forward residual model path')
    parser.add_argument(
        '--upsample-model',
        default='models/upsample.compilation/compilation.bmodel',
        help='Upsample model path')
    args = parser.parse_args()
    return args

def main():
    logger = init_logger()
    args = parse_args()

    clips = os.listdir(args.lrdir)
    results = [[] for i in clips]
    index = 0
    cv = Condition()
    def result_callback(out, info):
        img = tensor2img(torch.from_numpy(out))
        fn = os.path.basename(info['filename'])
        gtfn = os.path.join(args.gtdir, info['clip'], fn)
        gt = cv2.imread(gtfn)
        results[info['clip_index']].append([
            psnr(img, gt, convert_to='y'),
            ssim(img, gt, convert_to='y')])
        nonlocal index
        index += 1
        with cv:
            cv.notify()
    basicvsr = BasicVSR(
        args.spynet_model,
        args.backward_residual_model,
        args.forward_residual_model,
        args.upsample_model,
        result_callback,
        logger)
    import time
    start = time.time()
    total_lr_num = 0
    for i, clip in enumerate(clips):
        lrs = get_image_list(os.path.join(args.lrdir, clip))
        total_lr_num += len(lrs)
        basicvsr.put(lrs, clip=clip, clip_index=i)
        with cv:
            while index < total_lr_num:
                cv.wait()
    basicvsr.join()
    latency = time.time() - start
    logger.info('{:.2f}ms per frame'.format(latency * 1000 / total_lr_num))
    psnr_clip_avgs = []
    ssim_clip_avgs = []
    for i, clip in enumerate(clips):
        avg_psnr = np.mean([r[0] for r in results[i]])
        avg_ssim = np.mean([r[1] for r in results[i]])
        psnr_clip_avgs.append(avg_psnr)
        ssim_clip_avgs.append(avg_ssim)
        logger.info('{} psnr {:.4f}, ssim {:.4f}'.format(clip, avg_psnr, avg_ssim))
    psnr_avg = np.mean(psnr_clip_avgs)
    ssim_avg = np.mean(ssim_clip_avgs)
    logger.info('total psnr {:.4f}, ssim {:.4f}'.format(psnr_avg, ssim_avg))

if __name__ == '__main__':
    main()
