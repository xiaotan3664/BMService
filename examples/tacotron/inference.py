# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

from tacotron2.text import text_to_sequence
import models
import torch
import argparse
import os
import numpy as np
from scipy.io.wavfile import write
import matplotlib
import matplotlib.pyplot as plt

import sys

import time
import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

from waveglow.denoiser import Denoiser
from tacotron2_common.utils import get_mask_from_lengths

def init_decoder_inputs(memory, processed_memory, memory_lengths):

    device = memory.device
    dtype = memory.dtype
    bs = memory.size(0)
    seq_len = memory.size(1)
    attention_rnn_dim = 1024
    decoder_rnn_dim = 1024
    encoder_embedding_dim = 512
    n_mel_channels = 80

    attention_hidden = torch.zeros(bs, attention_rnn_dim, device=device, dtype=dtype)
    attention_cell = torch.zeros(bs, attention_rnn_dim, device=device, dtype=dtype)
    decoder_hidden = torch.zeros(bs, decoder_rnn_dim, device=device, dtype=dtype)
    decoder_cell = torch.zeros(bs, decoder_rnn_dim, device=device, dtype=dtype)
    attention_weights = torch.zeros(bs, seq_len, device=device, dtype=dtype)
    attention_weights_cum = torch.zeros(bs, seq_len, device=device, dtype=dtype)
    attention_context = torch.zeros(bs, encoder_embedding_dim, device=device, dtype=dtype)
    mask = get_mask_from_lengths(memory_lengths).to(device)
    decoder_input = torch.zeros(bs, n_mel_channels, device=device, dtype=dtype)

    return (decoder_input, attention_hidden, attention_cell, decoder_hidden,
            decoder_cell, attention_weights, attention_weights_cum,
            attention_context, memory, processed_memory, mask)

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='full path to the input text (phareses separated by new line)')
    parser.add_argument('-o', '--output', required=True,
                        help='output folder to save audio (file per phrase)')
    parser.add_argument('--suffix', type=str, default="", help="output filename suffix")
    parser.add_argument('--tacotron2', type=str,
                        help='full path to the Tacotron2 model checkpoint file')
    parser.add_argument('--waveglow', type=str,
                        help='full path to the WaveGlow model checkpoint file')
    parser.add_argument('-s', '--sigma-infer', default=0.9, type=float)
    parser.add_argument('-d', '--denoising-strength', default=0.01, type=float)
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')

    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument('--fp16', action='store_true',
                        help='Run inference with mixed precision')
    run_mode.add_argument('--cpu', action='store_true',
                        help='Run inference on CPU')

    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='Filename for logging')
    parser.add_argument('--include-warmup', action='store_true',
                        help='Include warmup')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')

    return parser


def checkpoint_from_distributed(state_dict):
    """
    Checks whether checkpoint was generated by DistributedDataParallel. DDP
    wraps model in additional "module.", it needs to be unwrapped for single
    GPU inference.
    :param state_dict: model's state dict
    """
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret


def unwrap_distributed(state_dict):
    """
    Unwraps model from DistributedDataParallel.
    DDP wraps model in additional "module.", it needs to be removed for single
    GPU inference.
    :param state_dict: model's state dict
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict


def load_and_setup_model(model_name, parser, checkpoint, fp16_run, cpu_run, forward_is_infer=False):
    model_parser = models.model_parser(model_name, parser, add_help=False)
    model_args, _ = model_parser.parse_known_args()

    model_config = models.get_model_config(model_name, model_args)
    model = models.get_model(model_name, model_config, cpu_run=cpu_run,
                             forward_is_infer=forward_is_infer)

    if checkpoint is not None:
        if cpu_run:
            state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))['state_dict']
        else:
            state_dict = torch.load(checkpoint)['state_dict']
        if checkpoint_from_distributed(state_dict):
            state_dict = unwrap_distributed(state_dict)

        model.load_state_dict(state_dict)

    if model_name == "WaveGlow":
        model = model.remove_weightnorm(model)

    model.eval()

    if fp16_run:
        model.half()

    return model


# taken from tacotron2/data_function.py:TextMelCollate.__call__
def pad_sequences(batch):
    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(x) for x in batch]),
        dim=0, descending=True)
    max_input_len = input_lengths[0]

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
        text = batch[ids_sorted_decreasing[i]]
        text_padded[i, :text.size(0)] = text

    return text_padded, input_lengths


def prepare_input_sequence(texts, cpu_run=True):

    d = []
    for i,text in enumerate(texts):
        d.append(torch.IntTensor(
            text_to_sequence(text, ['english_cleaners'])[:]))

    text_padded, input_lengths = pad_sequences(d)
    if not cpu_run:
        text_padded = text_padded.cuda().long()
        input_lengths = input_lengths.cuda().long()
    else:
        text_padded = text_padded.long()
        input_lengths = input_lengths.long()

    return text_padded, input_lengths


class MeasureTime():
    def __init__(self, measurements, key, cpu_run=True):
        self.measurements = measurements
        self.key = key
        self.cpu_run = cpu_run

    def __enter__(self):
        if not self.cpu_run:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not self.cpu_run:
            torch.cuda.synchronize()
        self.measurements[self.key] = time.perf_counter() - self.t0


def main():
    """
    Launches text to speech (inference).
    Inference is executed on a single GPU or CPU.
    """
    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    log_file = os.path.join(args.output, args.log_file)
    DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT, log_file),
                            StdOutBackend(Verbosity.VERBOSE)])
    for k,v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k:v})
    DLLogger.log(step="PARAMETER", data={'model_name':'Tacotron2_PyT'})

    tacotron2 = load_and_setup_model('Tacotron2', parser, args.tacotron2,
                                     args.fp16, args.cpu, forward_is_infer=True)
    waveglow = load_and_setup_model('WaveGlow', parser, args.waveglow,
                                    args.fp16, args.cpu, forward_is_infer=True)
    denoiser = Denoiser(waveglow)
    if not args.cpu:
        denoiser.cuda()

    jitted_tacotron2 = torch.jit.script(tacotron2)

    texts = []
    try:
        f = open(args.input, 'r')
        texts = f.readlines()
    except:
        print("Could not read file")
        sys.exit(1)

    if args.include_warmup:
        sequence = torch.randint(low=0, high=148, size=(1,50)).long()
        input_lengths = torch.IntTensor([sequence.size(1)]).long()
        if not args.cpu:
            sequence = sequence.cuda()
            input_lengths = input_lengths.cuda()
        for i in range(3):
            with torch.no_grad():
                mel, mel_lengths, _ = jitted_tacotron2(sequence, input_lengths)
                _ = waveglow(mel)

    measurements = {}

    sequences_padded, input_lengths = prepare_input_sequence(texts, args.cpu)

    with torch.no_grad(), MeasureTime(measurements, "tacotron2_time", args.cpu):
        mel, mel_lengths, alignments = jitted_tacotron2(sequences_padded, input_lengths)

    with torch.no_grad(), MeasureTime(measurements, "waveglow_time", args.cpu):
        audios = waveglow(mel, sigma=args.sigma_infer)
        audios = audios.float()
    with torch.no_grad(), MeasureTime(measurements, "denoiser_time", args.cpu):
        audios = denoiser(audios, strength=args.denoising_strength).squeeze(1)

    print("Stopping after",mel.size(2),"decoder steps")
    tacotron2_infer_perf = mel.size(0)*mel.size(2)/measurements['tacotron2_time']
    waveglow_infer_perf = audios.size(0)*audios.size(1)/measurements['waveglow_time']

    DLLogger.log(step=0, data={"tacotron2_items_per_sec": tacotron2_infer_perf})
    DLLogger.log(step=0, data={"tacotron2_latency": measurements['tacotron2_time']})
    DLLogger.log(step=0, data={"waveglow_items_per_sec": waveglow_infer_perf})
    DLLogger.log(step=0, data={"waveglow_latency": measurements['waveglow_time']})
    DLLogger.log(step=0, data={"denoiser_latency": measurements['denoiser_time']})
    DLLogger.log(step=0, data={"latency": (measurements['tacotron2_time']+measurements['waveglow_time']+measurements['denoiser_time'])})

    for i, audio in enumerate(audios):

        plt.imshow(alignments[i].float().data.cpu().numpy().T, aspect="auto", origin="lower")
        figure_path = os.path.join(args.output,"alignment_"+str(i)+args.suffix+".png")
        plt.savefig(figure_path)

        audio = audio[:mel_lengths[i]*args.stft_hop_length]
        audio = audio/torch.max(torch.abs(audio))
        audio_path = os.path.join(args.output,"audio_"+str(i)+args.suffix+".wav")
        write(audio_path, args.sampling_rate, audio.cpu().numpy())

    DLLogger.flush()

if __name__ == '__main__':
    main()