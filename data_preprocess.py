import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from configuration import get_config
from utils import keyword_spot

config = get_config()   # get arguments from parser

# downloaded dataset path
audio_path= r'C:\Users\LG\Documents\DataSets\Vctk\data'                                          # utterance dataset
clean_path = r'C:\Users\LG\Documents\Deep_learning\speaker_vertification\clean_testset_wav'  # clean dataset
noisy_path = r'C:\Users\LG\Documents\Deep_learning\speaker_vertification\noisy_testset_wav'  # noisy dataset


def extract_noise():
    """ Extract noise and save the spectrogram (as numpy array in config.noise_path)
        Need: paired clean and noisy data set
    """
    print("start noise extraction!")
    os.makedirs(config.noise_path, exist_ok=True)              # make folder to save noise file
    total = len(os.listdir(clean_path))                        # total length of audio files
    batch_frames = config.N * config.M * config.tdsv_frame     # TD-SV frame number of each batch
    stacked_noise = []
    stacked_len = 0
    k = 0
    for i, path in enumerate(os.listdir(clean_path)):
        clean, sr = librosa.core.load(os.path.join(clean_path, path), sr=8000)  # load clean audio
        noisy, _ = librosa.core.load(os.path.join(noisy_path, path), sr=sr)     # load noisy audio
        noise = clean - noisy       # get noise audio by subtract clean voice from the noisy audio
        S = librosa.core.stft(y=noise, n_fft=config.nfft,
                              win_length=int(config.window * sr), hop_length=int(config.hop * sr))   # perform STFT
        stacked_noise.append(S)
        stacked_len += S.shape[1]

        if i % 100 == 0:
            print("%d processing..." % i)

        if stacked_len < batch_frames:   # if noise frames is short than batch frames, then continue to stack the noise
            continue

        stacked_noise = np.concatenate(stacked_noise, axis=1)[:,:batch_frames]          # concat noise and slice
        np.save(os.path.join(config.noise_path, "noise_%d.npy" % k), stacked_noise)   # save spectrogram as numpy file
        print(" %dth file saved" % k, stacked_noise.shape)
        stacked_noise = []     # reset list
        stacked_len = 0
        k += 1

    print("noise extraction is end! %d noise files" % k)


def save_spectrogram_tdsv():
    """ Select text specific utterance and perform STFT with the audio file.
        Audio spectrogram files are divided as train set and test set and saved as numpy file. 
        Need : utterance data set (VTCK)
    """
    print("start text dependent utterance selection")
    os.makedirs(config.train_path, exist_ok=True)   # make folder to save train file
    os.makedirs(config.test_path, exist_ok=True)    # make folder to save test file

    utterances_spec = []
    for folder in os.listdir(audio_path):
        utter_path= os.path.join(audio_path, folder, os.listdir(os.path.join(audio_path, folder))[0])
        if os.path.splitext(os.path.basename(utter_path))[0][-3:] != '001':  # if the text utterance doesn't exist pass
            print(os.path.basename(utter_path)[:4], "001 file doesn't exist")
            continue

        utter, sr = librosa.core.load(utter_path, config.sr)               # load the utterance audio
        utter_trim, index = librosa.effects.trim(utter, top_db=14)         # trim the beginning and end blank
        if utter_trim.shape[0]/sr <= config.hop*(config.tdsv_frame+2):     # if trimmed file is too short, then pass
            print(os.path.basename(utter_path), "voice trim fail")
            continue

        S = librosa.core.stft(y=utter_trim, n_fft=config.nfft,
                              win_length=int(config.window * sr), hop_length=int(config.hop * sr))  # perform STFT
        S = keyword_spot(S)          # keyword spot (for now, just slice last 80 frames which contains "Call Stella")
        utterances_spec.append(S)    # make spectrograms list

    utterances_spec = np.array(utterances_spec)  # list to numpy array
    np.random.shuffle(utterances_spec)           # shuffle spectrogram (by person)
    total_num = utterances_spec.shape[0]
    train_num = (total_num//10)*9                # split total data 90% train and 10% test
    print("selection is end")
    print("total utterances number : %d"%total_num, ", shape : ", utterances_spec.shape)
    print("train : %d, test : %d"%(train_num, total_num- train_num))
    np.save(os.path.join(config.train_path, "train.npy"), utterances_spec[:train_num])  # save spectrogram as numpy file
    np.save(os.path.join(config.test_path, "test.npy"), utterances_spec[train_num:])


def save_spectrogram_tisv():
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved. 
        Need : utterance data set (VTCK)
    """
    print("start text independent utterance feature extraction")
    os.makedirs(config.train_path, exist_ok=True)   # make folder to save train file
    os.makedirs(config.test_path, exist_ok=True)    # make folder to save test file

    utter_min_len = (config.tisv_frame * config.hop + config.window) * config.sr    # lower bound of utterance length
    total_speaker_num = len(os.listdir(audio_path))
    train_speaker_num= (total_speaker_num//10)*9            # split total data 90% train and 10% test
    print("total speaker number : %d"%total_speaker_num)
    print("train : %d, test : %d"%(train_speaker_num, total_speaker_num-train_speaker_num))
    for i, folder in enumerate(os.listdir(audio_path)):
        speaker_path = os.path.join(audio_path, folder)     # path of each speaker
        print("%dth speaker processing..."%i)
        utterances_spec = []
        k=0
        for utter_name in os.listdir(speaker_path):
            utter_path = os.path.join(speaker_path, utter_name)         # path of each utterance
            utter, sr = librosa.core.load(utter_path, config.sr)        # load utterance audio
            intervals = librosa.effects.split(utter, top_db=20)         # voice activity detection
            for interval in intervals:
                if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
                    utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
                    S = librosa.core.stft(y=utter_part, n_fft=config.nfft,
                                          win_length=int(config.window * sr), hop_length=int(config.hop * sr))
                    S = np.abs(S) ** 2
                    mel_basis = librosa.filters.mel(sr=config.sr, n_fft=config.nfft, n_mels=40)
                    S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances

                    utterances_spec.append(S[:, :config.tisv_frame])    # first 180 frames of partial utterance
                    utterances_spec.append(S[:, -config.tisv_frame:])   # last 180 frames of partial utterance

        utterances_spec = np.array(utterances_spec)
        print(utterances_spec.shape)
        if i<train_speaker_num:      # save spectrogram as numpy file
            np.save(os.path.join(config.train_path, "speaker%d.npy"%i), utterances_spec)
        else:
            np.save(os.path.join(config.test_path, "speaker%d.npy"%(i-train_speaker_num)), utterances_spec)


if __name__ == "__main__":
    extract_noise()
    if config.tdsv:
        save_spectrogram_tdsv()
    else:
        save_spectrogram_tisv()