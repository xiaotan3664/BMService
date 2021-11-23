# *****************************************************************************
#  Copyright (c) 2021, SOPHGO CORPORATION.  All rights reserved.
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

import bmservice
import os
import sys
from functools import reduce
import numpy as np
from scipy.io.wavfile import write
import torch
import argparse

from tacotron2_common.utils import get_mask_from_lengths
from tacotron2.text import text_to_sequence
from inference import MeasureTime, pad_sequences

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='full path to the input text (phareses separated by new line)')
    parser.add_argument('-o', '--output', required=True,
                        help='output folder to save audio (file per phrase)')
    parser.add_argument('--encoder', type=str, default='./encoder_compilation/compilation.bmodel',
                        help='full path to the Encoder bmodel')
    parser.add_argument('--decoder', type=str, default='./decoder_compilation/compilation.bmodel',
                        help='full path to the DecoderIter bmodel')
    parser.add_argument('--postnet', type=str, default='./postnet_compilation/compilation.bmodel',
                        help='full path to the Postnet bmodel')
    parser.add_argument('--waveglow', type=str, default='./waveglow_compilation/compilation.bmodel',
                        help='full path to the WaveGlow bmodel')
    parser.add_argument('--dump-dir', type=str, default='',
                        help='path to dump blobs')
    parser.add_argument('--mel-file', type=str, default='',
                        help='path to reference mel .npy')
    parser.add_argument('-d', '--denoising-strength', default=0.01, type=float)
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Infer batch size')
    parser.add_argument('--threads', type=int, required=True,
                        help='Infer threads')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')

    return parser

def prepare_input_sequences(texts, batch_size):

    input_list = []
    for i_start in range(0, len(texts), batch_size):
        d = []
        for text in texts[i_start:i_start+batch_size]:
            d.append(torch.IntTensor(
                text_to_sequence(text, ['english_cleaners'])[:]))

        # Pad to align batch size
        while len(d) < batch_size:
            d.append(d[-1])

        text_padded, input_lengths = pad_sequences(d)
        text_padded = text_padded.to(torch.int32).numpy()
        input_lengths = input_lengths.to(torch.int32).numpy()
        input_list.append((text_padded, input_lengths))

    return input_list

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def infer_tacotron2(encoder, decoder_iter, postnet,
                    input_list, measurements, dump_dir=None):
    """
    input_list contains thread_num * batch_num sequences
    """

    task_map = dict()
    for idx, input_data in enumerate(input_list):
        if dump_dir:
            np.save('input_{:02d}.npy'.format(idx), input_data[0])
        task_id = encoder.put(*input_data)
        task_map[task_id] = idx
    memory_list = [None for i in input_list]
    processed_memory_list = [None for i in input_list]
    for i in range(len(input_list)):
        task_id, values, ok = encoder.get()
        if not ok:
            raise RuntimeError('encoder failed')
        idx = task_map[task_id]
        memory, processed_memory, lens = values
        if dump_dir:
            np.save('memory_{:02d}.npy'.format(idx), memory)
        memory_list[idx] = memory
        processed_memory_list[idx] = processed_memory
        if dump_dir:
            np.save('processed_memory_{:02d}.npy'.format(idx), processed_memory)
    print('encoding done')

    n_mel_channels = 80
    attension_rnn_dim = 1024
    gate_threshold = 0.5
    max_decoder_steps = 1664
    mel_list = []
    attension_hidden_list = []
    attension_cell_list = []
    decoder_hidden_list = []
    decoder_cell_list = []
    attension_weights_list = []
    attension_weights_cum_list = []
    attension_context_list = []
    mask_list = []
    def decoder_put(idx):
        task_id = decoder_iter.put(
            mel_list[idx],
            attension_context_list[idx],
            attension_hidden_list[idx],
            attension_cell_list[idx],
            attension_weights_list[idx],
            attension_weights_cum_list[idx],
            processed_memory_list[idx],
            mask_list[idx],
            memory_list[idx],
            decoder_hidden_list[idx],
            decoder_cell_list[idx])
        task_map[task_id] = idx
    task_map.clear()
    for idx, input_data in enumerate(input_list):
        sequences, lengths = input_data
        mask = get_mask_from_lengths(torch.from_numpy(lengths)).numpy().astype(np.int32)
        batch_size = mask.shape[0]
        seq_len = memory_list[idx].shape[1]
        encoder_embedding_dim = memory_list[idx].shape[2]
        dtype = memory_list[idx].dtype
        mask_list.append(mask)
        mel_list.append(np.zeros((batch_size, n_mel_channels), dtype=dtype))
        attension_hidden_list.append(np.zeros((batch_size, attension_rnn_dim), dtype=dtype))
        attension_cell_list.append(np.zeros((batch_size, attension_rnn_dim), dtype=dtype))
        decoder_hidden_list.append(np.zeros((batch_size, attension_rnn_dim), dtype=dtype))
        decoder_cell_list.append(np.zeros((batch_size, attension_rnn_dim), dtype=dtype))
        attension_weights_list.append(np.zeros((batch_size, seq_len), dtype=dtype))
        attension_weights_cum_list.append(np.zeros((batch_size, seq_len), dtype=dtype))
        attension_context_list.append(np.zeros((batch_size, encoder_embedding_dim), dtype=dtype))
        decoder_put(idx)

    finish_list = [False for i in input_list]
    decoder_out_list = [None for i in input_list]
    while not all(finish_list):
        task_id, values, ok = decoder_iter.get()
        if not ok:
            raise RuntimeError('decoder failed')
        idx = task_map[task_id]
        gate_prediction = sigmoid(values[1])
        not_finished = np.squeeze((gate_prediction <= gate_threshold).astype(np.int32), axis=1)
        iter_index = 0
        reach_max_iters = False
        mel_output = np.expand_dims(values[0], axis=2)
        if decoder_out_list[idx] is None:
            # first iter
            decoder_out_list[idx] = [not_finished, mel_output]
        else:
            decoder_out = decoder_out_list[idx]
            iter_index = np.amax(decoder_out[0])
            decoder_out[0] += not_finished
            reach_max_iters = iter_index + 1 >= max_decoder_steps
            decoder_out[1] = np.concatenate(
                (decoder_out[1], mel_output),
                axis=2)
        if dump_dir:
            np.save(
                os.path.join(dump_dir, '{}.npy'.format(iter_index)),
                mel_output)
        if reach_max_iters or not np.any(not_finished):
            finish_list[idx] = True
        else:
            mel_list[idx] = values[0]
            attension_hidden_list[idx] = values[2]
            attension_cell_list[idx] = values[3]
            decoder_hidden_list[idx] = values[4]
            decoder_cell_list[idx] = values[5]
            attension_weights_list[idx] = values[6]
            attension_weights_cum_list[idx] = values[7]
            attension_context_list[idx] = values[8]
            decoder_put(idx)

    print('decoding done')

    task_map.clear()
    for idx, (mel_lengths, mel_outputs) in enumerate(decoder_out_list):
        if dump_dir:
            np.save('mel_{:02d}.npy'.format(idx), mel_outputs)
        task_id = postnet.put(mel_outputs)
        task_map[task_id] = idx
    for i in decoder_out_list:
        task_id, values, ok = postnet.get()
        if not ok:
            raise RuntimeError('decoder failed')
        if dump_dir:
            np.save('post_mel_{:02d}.npy'.format(idx), values[0])
        decoder_out_list[idx][1] = values[0]

    print('postnet done')

    return decoder_out_list

def infer_waveglow(waveglow, input_list, measurements, mel_file=''):
    """
    input_list contains thread_num * batch_num sequences
    """

    print('waveglow start')

    task_map = dict()
    stride = 256
    n_group = 8
    mel_step = waveglow.get_input_info()['mel'][-1]
    audios = []
    for idx, (mel_lengths, mel) in enumerate(input_list):
        batch_size, mel_channels, mel_size = mel.shape
        z_size = (mel_size * stride) // n_group
        z = np.random.randn(batch_size, n_group, z_size).astype(np.float32)
        if mel_file:
            mel = np.load('pt_post_mel.npy')
        idx_audios = []
        for i in range(0, mel_size, mel_step):
            submel = mel[:, :, i:i+mel_step]
            subz = z[:, :, i*stride//n_group:(i+mel_step)*stride//n_group]
            task_id = waveglow.put(submel, subz)
            task_map[task_id] = (idx, i // mel_step)
            idx_audios.append(None)
        audios.append(idx_audios)
    for i in task_map:
        task_id, values, ok = waveglow.get()
        if not ok:
            raise RuntimeError('waveglow failed')
        idx, sub_idx = task_map[task_id]
        audios[idx][sub_idx] = values[0]
    for i, idx_audios in enumerate(audios):
        audios[i] = np.concatenate(idx_audios, axis=1)

    print('waveglow done')

    return audios

def test_waveglow():
    import sys
    if len(sys.argv) != 3:
        print(sys.argv[0], '<waveglow.bmodel> <mel.npy>')
        return
    model_path, mel_path = sys.argv[1:]
    mel = np.load(mel_path)
    waveglow = bmservice.BMService(model_path)
    mel_size = mel.shape[-1]
    mel_lengths = np.array((mel_size,), dtype=np.int32)
    audios = infer_waveglow(waveglow, [[mel_lengths, mel]], {}, mel_file='')
    for idx, audio in enumerate(audios):
        audio = torch.from_numpy(np.squeeze(audio, axis=0))
        stft_hop_length = 256
        sample_rate = 22050
        audio = audio[:mel_size*stft_hop_length]
        audio = audio/torch.max(torch.abs(audio))
        audio_path = "test_waveglow_{}.wav".format(idx)
        write(audio_path, sample_rate, audio.cpu().numpy())

def main():

    parser = argparse.ArgumentParser(
        description='BMService Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    texts = []
    try:
        f = open(args.input, 'r', encoding='utf-8')
        texts = [line.split('|')[-1] for line in f.readlines()][:10]
        #texts = f.readlines()[:1]
    except:
        print("Could not read file")
        sys.exit(1)

    measurements = {}

    input_list = prepare_input_sequences(texts, args.batch_size)

    encoder = bmservice.BMService(args.encoder)
    decoder_iter = bmservice.BMService(args.decoder)
    postnet = bmservice.BMService(args.postnet)
    waveglow = bmservice.BMService(args.waveglow)

    for i_start in range(0, len(input_list), args.threads):
        iter_input = input_list[i_start:i_start+args.threads]
        iter_output = infer_tacotron2(
            encoder, decoder_iter, postnet,
            iter_input, measurements, args.dump_dir)
        mel_lengths = reduce(
            lambda acc, l: acc + l,
            (out[0].tolist() for out in iter_output), [])
        audios = infer_waveglow(waveglow, iter_output, measurements, args.mel_file)
        if args.dump_dir:
            np.save('audios.npy', audios)
        audios = reduce(lambda acc, arr: acc + list(arr), audios, [])

        for i, audio in enumerate(audios):
            audio = torch.from_numpy(audio)
            audio = audio[:mel_lengths[i]*args.stft_hop_length]
            audio = audio/torch.max(torch.abs(audio))
            audio_path = os.path.join(args.output, "audio_"+str(i)+"_bmrt.wav")
            write(audio_path, args.sampling_rate, audio.cpu().numpy())

if __name__ == "__main__":
    with torch.no_grad():
        main()
