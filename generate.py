from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer
import pandas as pd

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config_path = "./wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
model_path = "./wavtokenizer_medium_speech_320_24k.ckpt"

wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device)

wav, sr = torchaudio.load("./1.wav")
wav = convert_audio(wav, sr, 24000, 1) 
bandwidth_id = torch.tensor([0])
wav=wav.to(device)
_,discrete_code= wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
print(discrete_code)
discrete_code_np=discrete_code.cpu().numpy()