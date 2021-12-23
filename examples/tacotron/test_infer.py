# *****************************************************************************
#  Copyright (c) 2018, SOPHGO CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the SOPHGO CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL SOPHGO CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import sys
import os
from tacotron2.text import text_to_sequence
import bmservice
import models
import torch
import argparse
import numpy as np
from scipy.io.wavfile import write
from functools import reduce

from inference import MeasureTime
from tacotron2_common.utils import get_mask_from_lengths
from inference_bmservice import prepare_input_sequences, infer_tacotron2, infer_waveglow, sigmoid

import time
import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('--encoder', type=str, default='./encoder_compilation/compilation.bmodel',
                        help='full path to the Encoder bmodel')
    parser.add_argument('--decoder', type=str, default='./decoder_compilation/compilation.bmodel',
                        help='full path to the DecoderIter bmodel')
    parser.add_argument('--postnet', type=str, default='./postnet_compilation/compilation.bmodel',
                        help='full path to the Postnet bmodel')
    parser.add_argument('--waveglow', type=str, default='./waveglow_compilation/compilation.bmodel',
                        help='full path to the WaveGlow bmodel')
    parser.add_argument('-o', '--output', required=True,
                        help='output folder to save audio (file per phrase)')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('--num-iters', type=int, default=10,
                        help='Number of iterations')
    parser.add_argument('-il', '--input-length', type=int, default=128,
                        help='Input length')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Infer batch size')
    parser.add_argument('--max-decoder-steps', type=int, default=1664,
                        help='Max decoder steps')

    return parser

from queue import Queue
from threading import Thread

class Pipeline:

    def __init__(self, encoder_path, decoder_path, postnet_path, waveglow_path, pre_fn, post_fn, max_decoder_steps):
        self.max_decoder_steps = max_decoder_steps
        self.pre_fn = pre_fn
        self.post_fn = post_fn
        self.encoder = bmservice.BMService(encoder_path)
        self.decoder = bmservice.BMService(decoder_path)
        self.postnet = bmservice.BMService(postnet_path)
        self.waveglow = bmservice.BMService(waveglow_path)
        maxsize = 2
        self.encoder_q  = Queue(maxsize)
        self.encoder_map = dict()
        self.decoder_map = dict()
        self.postnet_map = dict()
        self.waveglow_map = dict()
        self.decoder_drain_flag = False

    def start(self):
        self.feed_encoder_t  = Thread(target=self.feed_encoder)
        self.feed_decoder_t  = Thread(target=self.feed_decoder)
        self.iter_decoder_t  = Thread(target=self.iter_decoder)
        self.feed_waveglow_t = Thread(target=self.feed_waveglow)
        self.post_t = Thread(target=self.post_process)
        self.feed_encoder_t.start()

    def start_thread(self, t):
        if t.is_alive():
            return
        t.start()

    def join_thread(self, t):
        if not t.is_alive():
            return
        t.join()

    def show(self):
        self.encoder.show()
        self.decoder.show()
        self.postnet.show()
        self.waveglow.show()

    def join(self):
        self.join_thread(self.feed_encoder_t)
        self.join_thread(self.feed_decoder_t)
        self.join_thread(self.iter_decoder_t)
        self.join_thread(self.feed_waveglow_t)
        self.join_thread(self.post_t)

    def feed_encoder(self):
        while True:
            text = self.pre_fn()
            if text is None:
                self.encoder.put()
                break
            sequences, lengths = prepare_input_sequences([text], 1)[0]
            perf_time = time.perf_counter()
            task_id = self.encoder.put(sequences, lengths)
            self.start_thread(self.feed_decoder_t)
            self.encoder_map[task_id] = dict(task_id=task_id, encode_start=perf_time, lengths=lengths)

    def decoder_put(self, task):
        return self.decoder.put(
            task['mel'],
            task['attension_context'],
            task['attension_hidden'],
            task['attension_cell'],
            task['attension_weights'],
            task['attension_weights_cum'],
            task['processed_memory'],
            task['mask'],
            task['memory'],
            task['decoder_hidden'],
            task['decoder_cell'])

    def feed_decoder(self):
        n_mel_channels = 80
        attension_rnn_dim = 1024

        while True:
            task_id, values, ok = self.encoder.get()
            if not ok:
                self.decoder_drain_flag = True
                break
            self.start_thread(self.iter_decoder_t)
            task = self.encoder_map.pop(task_id)
            memory, processed_memory, lens = values
            task['memory'] = memory
            task['processed_memory'] = processed_memory
            lengths = task['lengths']
            batch_size = lengths.shape[0]
            seq_len = memory.shape[1]
            encoder_embedding_dim = memory.shape[2]
            dtype = memory.dtype
            task['mask'] = get_mask_from_lengths(torch.from_numpy(lengths)).numpy().astype(np.int32)
            task['mel'] = np.zeros((batch_size, n_mel_channels), dtype=dtype)
            task['attension_hidden'] = np.zeros((batch_size, attension_rnn_dim), dtype=dtype)
            task['attension_cell'] = np.zeros((batch_size, attension_rnn_dim), dtype=dtype)
            task['decoder_hidden'] = np.zeros((batch_size, attension_rnn_dim), dtype=dtype)
            task['decoder_cell'] = np.zeros((batch_size, attension_rnn_dim), dtype=dtype)
            task['attension_weights'] = np.zeros((batch_size, seq_len), dtype=dtype)
            task['attension_weights_cum'] = np.zeros((batch_size, seq_len), dtype=dtype)
            task['attension_context'] = np.zeros((batch_size, encoder_embedding_dim), dtype=dtype)
            task['decode_start'] = time.perf_counter()
            task_id = self.decoder_put(task)
            self.decoder_map[task_id] = task

    def iter_decoder(self):
        gate_threshold = 0.5
        while True:
            task_id, values, ok = self.decoder.get()
            if not ok:
                break
            task = self.decoder_map.pop(task_id)
            gate_prediction = sigmoid(values[1])
            not_finished = np.squeeze((gate_prediction <= gate_threshold).astype(np.int32), axis=1)
            mel_output = np.expand_dims(values[0], axis=2)
            iter_index = 0
            reach_max_iters = False
            if 'mel_outputs' not in task:
                # first iter
                task['mel_outputs'] = mel_output
                task['mel_lengths'] = not_finished
            else:
                task['mel_lengths'] += not_finished
                iter_index = np.amax(task['mel_lengths'])
                reach_max_iters = iter_index >= self.max_decoder_steps
                task['mel_outputs'] = np.concatenate((task['mel_outputs'], mel_output), axis=2)

            if reach_max_iters or not np.any(not_finished):
                task_id = self.postnet.put(task['mel_outputs'])
                self.postnet_map[task_id] = task
                self.start_thread(self.feed_waveglow_t)
                if not self.decoder_map and self.decoder_drain_flag:
                    self.decoder.put()
                    self.postnet.put()
            else:
                task['mel'] = values[0]
                task['attension_hidden'] = values[2]
                task['attension_cell'] = values[3]
                task['decoder_hidden'] = values[4]
                task['decoder_cell'] = values[5]
                task['attension_weights'] = values[6]
                task['attension_weights_cum'] = values[7]
                task['attension_context'] = values[8]
                task_id = self.decoder_put(task)
                self.decoder_map[task_id] = task

    def infer_waveglow(self, mel):
        stride = 256
        n_group = 8
        mel_step = self.waveglow.get_input_info()['mel'][-1]
        batch_size, mel_channels, mel_size = mel.shape
        z_size = (mel_size * stride) // n_group
        z = np.random.randn(batch_size, n_group, z_size).astype(np.float32)
        task_ids = []
        for i in range(0, mel_size, mel_step):
            submel = mel[:, :, i:i+mel_step]
            subz = z[:, :, i*stride//n_group:(i+mel_step)*stride//n_group]
            task_id = self.waveglow.put(submel, subz)
            task_ids.append(task_id)
        return task_ids

    def feed_waveglow(self):
        while True:
            task_id, values, ok = self.postnet.get()
            if not ok:
                self.waveglow.put()
                break
            self.start_thread(self.post_t)
            task = self.postnet_map.pop(task_id)
            post_mel = values[0]
            task_ids = self.infer_waveglow(post_mel)
            task['waveglow_outs'] = [None for i in task_ids]
            task['mel_size'] = post_mel.shape[-1]
            for idx, task_id in enumerate(task_ids):
                self.waveglow_map[task_id] = (task, idx)

    def post_process(self):
        while True:
            task_id, values, ok = self.waveglow.get()
            if not ok:
                break
            (task, idx) = self.waveglow_map.pop(task_id)
            task['waveglow_outs'][idx] = values[0]
            if all(out is not None for out in task['waveglow_outs']):
                audio = np.concatenate(task['waveglow_outs'], axis=1)
                audio = torch.from_numpy(np.squeeze(audio, axis=0))
                stft_hop_length = 256
                audio = audio[:task['mel_size']*stft_hop_length]
                audio = audio/torch.max(torch.abs(audio))
                self.post_fn(audio.cpu().numpy())

def main():
    """
    Launches text to speech (inference).
    Inference is executed on a single GPU.
    """
    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 Inference')
    parser = parse_args(parser)
    args, unknown_args = parser.parse_known_args()

    text = "The forms of printed letters should be beautiful, and that their arrangement on the page should be reasonable and a help to the shapeliness of the letters themselves. The forms of printed letters should be beautiful, and that their arrangement on the page should be reasonable and a help to the shapeliness of the letters themselves."
    text = text[:args.input_length]
    iter_index = 0
    def gen_text():
        nonlocal iter_index
        if iter_index >= args.num_iters:
            return None
        iter_index += 1
        return text

    audio_index = 0
    def save_audio(audio):
        sample_rate = 22050
        nonlocal audio_index
        audio_path = os.path.join(args.output, 'test_infer_{:02d}.wav'.format(audio_index))
        write(audio_path, sample_rate, audio)
        audio_index += 1

    runner = Pipeline(
        encoder_path=args.encoder,
        decoder_path=args.decoder,
        postnet_path=args.postnet,
        waveglow_path=args.waveglow,
        pre_fn=gen_text,
        post_fn=save_audio,
        max_decoder_steps=args.max_decoder_steps)

    start = time.perf_counter()
    runner.start()
    runner.join()
    end = time.perf_counter()
    print('Average time cost {:.02f}ms'.format((end - start) * 1000 / args.num_iters))
    runner.show()

if __name__ == '__main__':
    main()
