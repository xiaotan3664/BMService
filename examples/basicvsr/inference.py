import argparse
import os
import cv2
import numpy as np
import bmservice
import torch
import torch.nn.functional as F
from flow_warp import flow_warp
import logging
from misc import tensor2img

def init_logger():
    global logger
    logging.basicConfig(
        format='[%(pathname)s:%(lineno)d] %(levelname)s - %(message)s',
        datefmt='%m-%d %H:%M:%S',
        level=logging.INFO)
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='BasicVSR Inference')
    parser.add_argument('imagedir', help='Image dir')
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
    parser.add_argument(
        '--dump-input',
        action='store_true',
        help='Dump input for calibration')
    args = parser.parse_args()
    return args

def get_image_list(path):
    filenames = os.listdir(path)
    filenames.sort()
    return [os.path.join(path, fn) for fn in filenames]

def up_align(v, base):
    return (v + base - 1) // base * base

def down_align(v, base):
    return v // base * base

class Net:
    def __init__(self, model_path, logger, dump_input):
        self.logger = logger
        self.dump_input = dump_input
        self.model_path = model_path
        self.service = bmservice.BMService(model_path)
        self.input_info = self.service.get_input_info()
        from threading import Thread, Condition
        self.worker = Thread(target=self.work)
        self.callback = None
        self.running = True
        self.worker.start()
        self.joinable = False
        self.join_cv = Condition()

        if self.dump_input:
            import ufw
            name = model_path.replace('/', '_')
            self.txn = []
            for key in self.input_info:
                fn = '{}-{}.lmdb'.format(name, key).strip('._')
                logger.info(fn)
                txn = ufw.io.LMDB_Dataset(fn)
                self.txn.append(txn)

    def __getattr__(self, key):
        return getattr(self.service, key)

    def work(self):
        try:
            while self.running:
                task_id, values, ok = self.service.get()
                if not ok:
                    break
                values = self.postprocess(task_id, values)
                if self.callback is None:
                    raise RuntimeError('callback is None but result arrived')
                ret = self.callback(task_id, values)
                if ret is None or ret:
                    with self.join_cv:
                        self.joinable = True
                        self.join_cv.notify()
        except Exception as err:
            self.logger.error('Excpetion {}'.format(err))
            raise err

    def postprocess(self, task_id, values):
        return values

    def set_callback(self, cb):
        self.callback = cb

    def join_thread(self, t):
        if not t.is_alive():
            return
        t.join()

    def join(self):
        with self.join_cv:
            while not self.joinable:
                self.join_cv.wait()
        self.service.put()
        self.join_thread(self.worker)
        if self.dump_input:
            for txn in self.txn:
                txn.close()

    def put(self, *tensors):
        if self.dump_input:
            for txn, tensor in zip(self.txn, tensors):
                txn.put(tensor)
        return self.service.put(*tensors)

class SPyNet(Net):
    def __init__(self, model_path, logger, dump_input):
        super().__init__(model_path, logger, dump_input)
        self.input_shape = list(self.input_info.values())[0]
        _, _, self.input_height, self.input_width = self.input_shape
        self.input_aspect_ratio = self.input_width / self.input_height

    def preprocess(self, image):
        return self.preprocess_dynamic(image)

    def preprocess_dynamic(self, image):
        if len(image.shape) == 4:
            _, _, h, w = image.shape
        else:
            h, w, _ = image.shape
        input_h, input_w = up_align(h, 32), up_align(w, 32)
        if input_h > self.input_height or input_w > self.input_width:
            factor = min(self.input_width / input_w, self.input_height / input_h)
            input_h = down_align(input_h * factor, 32)
            input_w = down_align(input_w * factor, 32)
        return F.interpolate(
            input=torch.from_numpy(image), size=(input_h, input_w),
            mode='bilinear', align_corners=False).numpy()

def normalize(image):
    t = image.transpose([2, 0, 1]).astype(np.float32) / 255.0
    return np.expand_dims(t, 0)

class BasicVSR:
    def __init__(
            self, spynet, backward_residual, forward_residual, upsample,
            result_callback, logger, dump_input=False):

        self.logger = logger
        self.spynet = SPyNet(spynet, logger, dump_input)
        self.spynet.set_callback(self.flow_callback)
        from collections import OrderedDict
        from threading import Lock
        self.backward_flow_task_id = OrderedDict()
        self.forward_flow_task_id = OrderedDict()

        self.backward_residual = Net(backward_residual, logger, dump_input)
        self.forward_residual = Net(forward_residual, logger, dump_input)
        self.upsample = Net(upsample, logger, dump_input)

        self.backward_residual.set_callback(self.backward_residual_callback)
        self.forward_residual.set_callback(self.forward_residual_callback)

        self.upsample.set_callback(self.upsample_callback)
        self.result_callback = result_callback

        self.backward_residual_task_id = dict()
        self.hbs = []
        self.backward_lock = Lock()
        self.forward_residual_task_id = dict()
        self.hfs = []
        self.forward_lock = Lock()

        self.upsample_lock = Lock()
        self.upsample_task_id = dict()

    mid_channels = 64

    def upsample_callback(self, task_id, values):
        out = values[0]
        info = self.upsample_task_id.pop(task_id)
        if not self.result_callback:
            self.logger.error('no result_callback')
            return
        self.result_callback(out, info)

    def start_upsample(self, idx):
        self.logger.debug(
            'idx: {}, hb: {}, hf: {}'.format(
                idx, bool(self.hbs[idx]), bool(self.hfs[idx])))
        if self.hbs[idx] is None:
            return
        if self.hfs[idx] is None:
            return
        normalized = self.hbs[idx]['normalized']
        info = self.hbs[idx]
        if id(self.hbs[idx]) != id(self.hfs[idx]):
            raise RuntimeError('invalid state')
        hb = self.hbs[idx]['backward_h']
        hf = self.hfs[idx]['forward_h']

        # Inject propagation features
        #hb = np.load('propfeats/{}/backward-feat-{:08d}.npy'.format(
        #    info['clip'], info['idx']))
        #hf = np.load('propfeats/{}/forward-feat-{:08d}.npy'.format(
        #    info['clip'], info['idx']))

        self.logger.debug('{} {}'.format('do upsample', idx))
        task_id = self.upsample.put(hb, hf, normalized)
        self.upsample_task_id[task_id] = info

    def start_residual(self, name, task_map, last_idx_fn, hs, net, next_map):
        # this method try to put flow result map to residual forwarding
        # residual module requires:
        # flow
        # last residual
        flow_key = '{}_flow'.format(name)
        h_key = '{}_h'.format(name)
        while True:
            # residual forward with hidden state
            # must in order
            if not task_map:
                return
            key = list(task_map.keys())[0]
            info = task_map[key]
            if info.get(flow_key) is None:
                break
            idx = info['idx']
            last_idx = last_idx_fn(idx)
            self.logger.debug('start residual {} idx: {}, last_idx: {}, {}'
                .format(name, idx, last_idx, bool(hs[last_idx])))
            if not hs[last_idx]:
                return
            h = hs[last_idx][h_key]
            if info.get('normalized') is None:
                info['normalized'] = normalize(info['lr'])
            lr = info['normalized']
            hs_tensor = torch.from_numpy(h)
            feat_prop = flow_warp(hs_tensor, info[flow_key]).numpy()
            task_id = net.put(lr, feat_prop)
            task_map.pop(key)
            next_map[task_id] = info

    def residual_callback(self, name, task_id, values, task_map, hs):
        h = values[0]
        info = task_map.pop(task_id)
        idx = info['idx']
        self.logger.debug('{} {} {}'.format(name, 'residual', idx))
        info['{}_h'.format(name)] = h
        with self.upsample_lock:
            # protect hbs & hfs
            hs[idx] = info
            self.start_upsample(idx)
        return info

    def backward_residual_callback(self, task_id, values):
        self.residual_callback(
            'backward',
            task_id, values,
            task_map=self.backward_residual_task_id,
            hs=self.hbs)
        with self.backward_lock:
            self.start_residual(
                'backward',
                self.backward_flow_task_id,
                lambda x: x + 1,
                self.hbs,
                self.backward_residual,
                self.backward_residual_task_id)
        return all(self.hbs)

    def forward_residual_callback(self, task_id, values):
        self.residual_callback(
            'forward',
            task_id, values,
            task_map=self.forward_residual_task_id,
            hs=self.hfs)
        with self.forward_lock:
            self.start_residual(
                'forward',
                self.forward_flow_task_id,
                lambda x: x - 1,
                self.hfs,
                self.forward_residual,
                self.forward_residual_task_id)
        return all(self.hfs)

    def flow_callback(self, task_id, values):
        flow = values[0]
        _, _, flow_height, flow_width = flow.shape
        def interpolate(flow, info):
            h, w, _ = info['lr'].shape
            flow_tensor = torch.from_numpy(flow)
            if w != flow_width or h != flow_height:
                flow_tensor = F.interpolate(
                    input=flow_tensor,
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False)
                flow_tensor[:, 0, :, :] *= w / flow_width
                flow_tensor[:, 1, :, :] *= h / flow_height
            return flow_tensor.permute(0, 2, 3, 1)
        if task_id in self.backward_flow_task_id:
            info = self.backward_flow_task_id[task_id]
            # Inject flow
            #flow = np.load('flows/{}/backward-{:08d}.npy'.format(
            #    info['clip'], info['idx']))

            info['backward_flow'] = interpolate(flow, info)
            self.logger.debug('backward flow {}'.format(info['idx']))
            with self.backward_lock:
                self.start_residual(
                    'backward',
                    self.backward_flow_task_id,
                    lambda x: x + 1,
                    self.hbs,
                    self.backward_residual,
                    self.backward_residual_task_id)
        elif task_id in self.forward_flow_task_id:
            info = self.forward_flow_task_id[task_id]
            # Inject flow
            #flow = np.load('flows/{}/forward-{:08d}.npy'.format(
            #    info['clip'], info['idx'] - 1))

            info['forward_flow'] = interpolate(flow, info)
            self.logger.debug('forward flow {}'.format(info['idx']))
            with self.forward_lock:
                self.start_residual(
                    'forward',
                    self.forward_flow_task_id,
                    lambda x: x - 1,
                    self.hfs,
                    self.forward_residual,
                    self.forward_residual_task_id)
        else:
            raise RuntimeError('invalid task')

        ret = all(
                'backward_flow' in info
                for info in self.backward_flow_task_id.values()) and \
              all(
                'forward_flow' in info
                for info in self.forward_flow_task_id.values())

        return ret

    def put(self, lr_files, **extra_info):
        def to_input_tensor(lr_fn):
            lr = cv2.imread(lr_fn)
            rgb = lr[:, :, ::-1]
            normalized = normalize(rgb)
            spynet_input = self.spynet.preprocess(normalized)
            return dict(
                filename=lr_fn, lr=lr,
                normalized=normalized,
                spynet_input=spynet_input,
                **extra_info)

        # forward & backward hidden state
        # also the propagation output of each lr
        self.hfs = [None for i in lr_files]
        self.hbs = [None for i in lr_files]

        lrs = [None for i in lr_files]
        lrs[0] = to_input_tensor(lr_files[0])
        lrs[0]['idx'] = 0
        lrs[-1] = to_input_tensor(lr_files[-1])
        lrs[-1]['idx'] = len(lr_files) - 1

        # start initial resfeat
        h, w, _ = lrs[0]['lr'].shape
        feat_shape = [1, self.mid_channels, h, w]
        zero_feat_prop = np.zeros(feat_shape, dtype=np.float32)
        # forward init
        task_id = self.forward_residual.put(
            lrs[0]['normalized'], zero_feat_prop)
        self.forward_residual_task_id[task_id] = lrs[0]
        # backward init
        task_id = self.backward_residual.put(
            lrs[0]['normalized'], zero_feat_prop)
        self.backward_residual_task_id[task_id] = lrs[-1]

        for i in range(len(lr_files) - 1):
            # forward
            prev_lr = lrs[i]
            curr_index = i + 1
            if lrs[curr_index] is None:
                lrs[curr_index] = to_input_tensor(lr_files[curr_index])
            curr_lr = lrs[curr_index]
            forward_task_id = self.spynet.put(
                curr_lr['spynet_input'], prev_lr['spynet_input'])
            curr_lr['idx'] = curr_index
            with self.forward_lock:
                self.forward_flow_task_id[forward_task_id] = curr_lr

            # backward
            next_lr = lrs[len(lr_files) - i - 1]
            curr_index = len(lr_files) - i - 2
            if lrs[curr_index] is None:
                lrs[curr_index] = to_input_tensor(lr_files[curr_index])
            curr_lr = lrs[curr_index]
            curr_lr['idx'] = curr_index
            backward_task_id = self.spynet.put(
                curr_lr['spynet_input'], next_lr['spynet_input'])
            with self.backward_lock:
                self.backward_flow_task_id[backward_task_id] = curr_lr

    def join(self):
        self.spynet.join()
        self.backward_residual.join()
        self.forward_residual.join()
        self.upsample.join()

def main():
    logger = init_logger()
    args = parse_args()
    def result_callback(out, info):
        idx = info['idx']
        logger.info('{} {}'.format(idx, out.shape))
        img = tensor2img(torch.from_numpy(out))
        cv2.imwrite('out/{}.png'.format(idx), img)
    basicvsr = BasicVSR(
        args.spynet_model,
        args.backward_residual_model,
        args.forward_residual_model,
        args.upsample_model,
        result_callback,
        logger,
        args.dump_input)
    lrs = get_image_list(args.imagedir)
    import time
    start = time.time()
    basicvsr.put(lrs)
    basicvsr.join()
    latency = time.time() - start
    logger.info('{:.2f}ms per frame'.format(latency * 1000 / len(lrs)))

if __name__ == '__main__':
    main()

