import numpy as np
import math
import pandas as pd
import librosa
import pyworld
import glob

def world_decompose(wav, fs, frame_period = 5.0):
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 71.0, f0_ceil = 800.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    return f0, timeaxis, sp, ap

def world_encode_spectral_envelop(sp, fs, dim = 24):
    # 获取MCEP
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)
    return coded_sp

def wav_padding(wav, sr, frame_period, multiple = 4):
    assert wav.ndim == 1
    num_frames = len(wav)
    num_frames_padded = int((np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) + 1) / multiple + 1) * multiple - 1) * (sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right), 'constant', constant_values = 0)
    return wav_padded

num_mceps = 24
sampling_rate = 16000
frame_period = 5.0


filename_list_1 = glob.glob(r"./happy/*.wav")
filename_list_1=sorted(filename_list_1, key=lambda name: int(name[13:13+6]))
filename_list_2 = glob.glob(r"./neutral2happy/VCEVC_CHNFEMALE_-results/*.wav")
filename_list_2=sorted(filename_list_2, key=lambda name: int(name[46:46+6]))

mcdsum = 0
rmsesum = 0
pccsum = 0

for i in range(min(50, len(filename_list_2))):
    song1 = filename_list_1[i]
    song2 = filename_list_2[i]
    print(song1)
    print(song2)
    # 重采样、使音频长度相等
    wav1, _ = librosa.load(song1, sr=sampling_rate, mono=True)
    wav1 = wav_padding(wav=wav1, sr=sampling_rate, frame_period=frame_period, multiple=4)
    wav2, _ = librosa.load(song2, sr=sampling_rate, mono=True)
    # 改变wav2长度
    length = wav1.shape[0] / sampling_rate
    wav2 = librosa.resample(wav2, sampling_rate, int((sampling_rate / (wav2.shape[0] / sampling_rate)) * length),
                            fix=True, scale=True)
    wav2 = wav_padding(wav=wav2, sr=sampling_rate, frame_period=frame_period, multiple=4)


    f01, timeaxis1, sp1, ap1 = world_decompose(wav=wav1, fs=sampling_rate, frame_period=frame_period)
    coded_sp1 = world_encode_spectral_envelop(sp=sp1, fs=sampling_rate, dim=num_mceps)
    f02, timeaxis2, sp2, ap2 = world_decompose(wav=wav2, fs=sampling_rate, frame_period=frame_period)
    coded_sp2 = world_encode_spectral_envelop(sp=sp2, fs=sampling_rate, dim=num_mceps)

    # MCEP
    a = coded_sp1
    b = coded_sp2
    print(coded_sp1.shape[0], coded_sp2.shape[0])
    minlength = min(coded_sp1.shape[0], coded_sp2.shape[0])

    templist = []
    for i in range(minlength):
        sum = 0
        for j in range(24):
            temp = (a[i][j] - b[i][j]) ** 2
            sum += temp
        mcd_single = (10 / math.log(10)) * math.sqrt(2 * sum)
        templist.append(mcd_single)
    mcd_wav = np.mean(templist)
    print('MCD', mcd_wav)

    # f0
    a = f01
    b = f02
    sum = 0
    for i in range(minlength):
        sum += (a[i] - b[i]) ** 2
    rmse = math.sqrt(sum / minlength)
    print('RMSE', rmse)
    a_s = pd.Series(a)
    b_s = pd.Series(b)
    pcc = b_s.corr(a_s, method='pearson')
    print('PCC', pcc)

    mcdsum += mcd_wav
    rmsesum += rmse
    pccsum += abs(pcc)

print('Result:')
print('MCD', mcdsum / min(50, len(filename_list_2)))
print('RMSE', rmsesum / min(50, len(filename_list_2)))
print('PCC', pccsum / min(50, len(filename_list_2)))
