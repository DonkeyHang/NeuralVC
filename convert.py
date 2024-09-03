import os
import argparse
import torch
import librosa
import time
from scipy.io.wavfile import write
from tqdm import tqdm
import soundfile as sf
import utils
import time
from models import HuBERT_NeuralDec_VITS
from mel_processing import mel_spectrogram_torch
import logging
from tqdm import tqdm
import numpy as np

from hubert.hubert.model import HubertSoft
from speaker_encoder.voice_encoder import SpeakerEncoder
logging.getLogger('numba').setLevel(logging.WARNING)

PROCESS_BUFFER_SIZE = 160*30
PROCESS_OVERLAP_SIZE = int(PROCESS_BUFFER_SIZE/2)
FRAME_320 = 320

def vorbis_window(n):
    indices = torch.arange(n, dtype=torch.float32) + 0.5
    n_double = torch.tensor(n, dtype=torch.float32)
    window = torch.sin((torch.pi / 2.0) * torch.pow(torch.sin(indices / n_double * torch.pi), 2.0))
    
    return window

class Parameters:
    def __init__(self):
        self.hpfile = "logs/neuralvc/config.json"
        self.ptfile = "logs/neuralvc/G_NeuralVC.pth"
        self.model_name = "hubert-neuraldec-vits"
        self.outdir = "output/temp"
        self.use_timestamp = False

def consume_prefix_in_state_dict_if_present(state_dict, prefix='module.'):
    """Strip the prefix in state_dict in place, if any."""
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix):]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)


def ut():
    args = Parameters()
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

        os.makedirs(args.outdir, exist_ok=True)

    hps = utils.get_hparams_from_file(args.hpfile)

    print("Loading model...")
    net_g = HuBERT_NeuralDec_VITS(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    _ = net_g.eval()

    print("Loading checkpoint...")
    _ = utils.load_checkpoint(args.ptfile, net_g, None, True)

    print("Loading hubert...")
    # hubert_git = torch.hub.load("bshall/hubert:main", f"hubert_soft").eval() 
    hubert = HubertSoft()
    state_dict = torch.load("hubert/ckpt/hubert-soft-35d9f29f.pt")
    consume_prefix_in_state_dict_if_present(state_dict['hubert'], "module.")  # 调整前缀
    hubert.load_state_dict(state_dict["hubert"])
    hubert.eval()

    if hps.model.use_spk:
        print("Loading speaker encoder...")
        smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')
        print("ok")

    def convert(src_list, tgt):
        tgtname = tgt.split("/")[-1].split(".")[0]
        wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
        if not os.path.exists(os.path.join(args.outdir, tgtname)):
            os.makedirs(os.path.join(args.outdir, tgtname))
        sf.write(os.path.join(args.outdir, tgtname, f"tgt_{tgtname}.wav"), wav_tgt, hps.data.sampling_rate)
        wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)#input：[115676] output:[108544]
        
        g_tgt = smodel.embed_utterance(wav_tgt)
        g_tgt = torch.from_numpy(g_tgt).unsqueeze(0)        #input:[], output:[256]
        for src in tqdm(src_list):
            srcname = src.split("/")[-1].split(".")[0]
            title = srcname + "-" + tgtname
            with torch.no_grad():
                wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
                sf.write(os.path.join(args.outdir, tgtname, f"src_{srcname}.wav"), wav_src, hps.data.sampling_rate)
                wav_src = torch.from_numpy(wav_src).unsqueeze(0).unsqueeze(0)
                c = hubert.units(wav_src)
                c = c.transpose(1,2)                        #output:[1,256,_]
                audio = net_g.infer(c, g=g_tgt)
                audio = audio[0][0].data.cpu().float().numpy()
                write(os.path.join(args.outdir, tgtname, f"{title}.wav"), hps.data.sampling_rate, audio)




    tgt1 = "/Users/donkeyddddd/Documents/Rx_projects/git_projects/NeuralVC/wav/dolly_16k.wav"
    src_list1 = ["/Users/donkeyddddd/Documents/Rx_projects/git_projects/NeuralVC/wav/freddie_16k.wav"]
    convert(src_list1, tgt1)


def ut_2():
    import soundfile as sf

    audio1,sr1 = sf.read("/Users/donkeyddddd/Documents/Rx_projects/git_projects/NeuralVC/wav/dolly.wav")
    audio2,sr2 = sf.read("/Users/donkeyddddd/Documents/Rx_projects/git_projects/NeuralVC/wav/dolly_16k.wav")

    xxx = 1


def ut_3():
    args = Parameters()
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

        os.makedirs(args.outdir, exist_ok=True)

    hps = utils.get_hparams_from_file(args.hpfile)

    print("Loading model...")
    net_g = HuBERT_NeuralDec_VITS(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    _ = net_g.eval()

    print("Loading checkpoint...")
    _ = utils.load_checkpoint(args.ptfile, net_g, None, True)

    print("Loading hubert...")
    hubert = HubertSoft()
    state_dict = torch.load("hubert/ckpt/hubert-soft-35d9f29f.pt")
    consume_prefix_in_state_dict_if_present(state_dict['hubert'], "module.")
    hubert.load_state_dict(state_dict["hubert"])
    hubert.eval()

    if hps.model.use_spk:
        print("Loading speaker encoder...")
        smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')
        print("ok")

    def convert(src_list, tgt):
        tgtname = tgt.split("/")[-1].split(".")[0]
        wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
        if not os.path.exists(os.path.join(args.outdir, tgtname)):
            os.makedirs(os.path.join(args.outdir, tgtname))
        sf.write(os.path.join(args.outdir, tgtname, f"tgt_{tgtname}.wav"), wav_tgt, hps.data.sampling_rate)
        wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)#input：[115676] output:[108544]
        
        g_tgt = smodel.embed_utterance(wav_tgt)
        g_tgt = torch.from_numpy(g_tgt).unsqueeze(0)        #input:[], output:[256]

        win = vorbis_window(PROCESS_BUFFER_SIZE).detach().numpy()
        cache_buf = np.zeros(int(PROCESS_BUFFER_SIZE/2))
        

        for src in tqdm(src_list):
            srcname = src.split("/")[-1].split(".")[0]
            title = srcname + "-" + tgtname
            wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
            sf.write(os.path.join(args.outdir, tgtname, f"src_{srcname}.wav"), wav_src, hps.data.sampling_rate)
            
            output_audio = wav_src * 0
            num_frame = wav_src.shape[0] // PROCESS_OVERLAP_SIZE - 1
            
            with torch.no_grad():
                for idx in tqdm(range(num_frame)):
                    audio_tmp = wav_src[idx*PROCESS_OVERLAP_SIZE:idx*PROCESS_OVERLAP_SIZE+PROCESS_BUFFER_SIZE]
                    tensor_tmp = torch.from_numpy(audio_tmp).unsqueeze(0).unsqueeze(0)
                    c = hubert.units(tensor_tmp)
                    c = c.transpose(1,2)                        #output:[1,256,_]
                    audio = net_g.infer(c, g=g_tgt)
                    audio = audio[0][0].data.cpu().float().numpy() * win
                    output_audio[idx*PROCESS_OVERLAP_SIZE:(idx+1)*PROCESS_OVERLAP_SIZE] = audio[:PROCESS_OVERLAP_SIZE].copy() + cache_buf
                    cache_buf = audio[PROCESS_OVERLAP_SIZE:].copy()
                
            write(os.path.join(args.outdir, tgtname, f"{title}.wav"), hps.data.sampling_rate, output_audio)




    tgt1 = "/Users/donkeyddddd/Documents/Rx_projects/git_projects/NeuralVC/wav/dolly_16k.wav"
    src_list1 = ["/Users/donkeyddddd/Documents/Rx_projects/git_projects/NeuralVC/wav/freddie_16k.wav"]
    convert(src_list1, tgt1)
    

if __name__=="__main__":
    # ut()
    # ut_2()
    ut_3()

    xxx = 1