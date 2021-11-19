import argparse
import os

from convert_tacotron22onnx import main as convert_tacotron
from convert_waveglow2onnx import main as convert_waveglow

def compile_bmodel():
    import bmneto
    bmneto.compile(
        model='./onnx_models/encoder.onnx',
        outdir='encoder_compilation',
        net_name='encoder',
        target='BM1684',
        dyn=True,
        input_names=['sequences', 'sequence_lengths'],
        shapes=[[1, 512], [1]],
        descs='[0,int64,0,148],[1,int64,512,513]')
    bmneto.compile(
        model='./onnx_models/decoder_iter.onnx',
        net_name='decoder_iter',
        outdir='decoder_compilation',
        dyn=True,
        target='BM1684',
        input_names=[
            'decoder_input', 'attention_hidden', 'attention_cell', 'decoder_hidden',
            'decoder_cell', 'attention_weights', 'attention_weights_cum',
            'attention_context', 'memory', 'processed_memory', 'mask'],
        shapes=[
            [1, 80], [1, 1024], [1, 1024], [1, 1024], [1, 1024],
            [1, 512], [1, 512], [1, 512], [1, 512, 512], [1, 512, 128], [1, 512]],
        descs='[10,bool,0,2]')
    bmneto.compile(
        model='./onnx_models/postnet.onnx',
        net_name='postnet',
        outdir='postnet_compilation',
        dyn=True,
        target='BM1684',
        input_names=['mel_outputs'],
        shapes=[[1, 80, 1664]],
        cmp=1)
    mel_size = 512
    stride = 256
    n_group = 8
    z_size = (mel_size * stride) // n_group
    bmneto.compile(
        model='./onnx_models/waveglow.onnx',
        net_name='waveglow',
        outdir='waveglow_compilation',
        dyn=False,
        target='BM1684',
        input_names=['mel', 'z'],
        shapes=[[1, 80, mel_size],[1, n_group, z_size]],
        cmp=True)

def main():
    import sys
    parser = argparse.ArgumentParser(description='Model conversion')
    parser.add_argument('action', choices=['all', 'onnx', 'bmodel'], help='convert .pt to .onnx, or .onnx to .bmodel, or both')
    parser.add_argument('--tacotron2', type=str,
                        help='full path to the Tacotron2 model checkpoint file')
    parser.add_argument('--waveglow', type=str, required=True,
                        help='full path to the WaveGlow model checkpoint file')
    args = parser.parse_args()
    if args.action != 'bmodel':
        onnx_out_path = 'onnx_models'
        if not os.path.exists(onnx_out_path):
            os.make_dirs(onnx_out_path)
        convert_tacotron()
        convert_waveglow()
    if args.action != 'onnx':
        compile_bmodel()

if __name__ == '__main__':
    main()

