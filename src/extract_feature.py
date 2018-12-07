'''
Copyright (c) 2017 Hai Pham, Rutgers University
http://www.cs.rutgers.edu/~hxp1/

This code is free to use for academic/research purpose.

'''
import cv2
import math
import pathlib as plb
import librosa
import csv
import numpy as np
import matplotlib.pyplot as plt


def write_csv(filename, data):
    with open(filename, 'w', newline="") as csvfile:
        writer = csv.writer(csvfile)
        for arow in data:
            writer.writerow(arow)

def get_fps(videofile):
    cap = cv2.VideoCapture(videofile)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return (nFrame, fps)

def extract_one_frame_data(data, curPosition, nFrameSize, nSamPerFrame):
    frameData = np.zeros(nFrameSize, dtype=np.float32)
    if curPosition < 0:
        # need padding
        startPos = -curPosition
        frameData[startPos:nFrameSize] = data[0:(curPosition+nFrameSize)]
    else:
        if curPosition >= data.size:
            pass
        elif curPosition+nFrameSize > data.size:
            n = data.size - curPosition
            frameData[:n] = data[curPosition:]
        else:
            frameData[:] = data[curPosition:(curPosition+nFrameSize)]
    nextPos = curPosition + nSamPerFrame
    return (frameData, nextPos)
        

def extract_one_file(videofile, audiofile):
    print (" --- " + videofile)
    ### return mfcc, fbank
    # get video FPS
    nFrames, fps = get_fps(videofile)
    # load audio
    data, sr = librosa.load(audiofile, sr=44100) # data is np.float32
    # number of audio samples per video frame
    nSamPerFrame = int(math.floor(float(sr) / fps))
    # number of samples per 0.025s
    n25sSam = int(math.ceil(float(sr) * 0.025))
    # number of sample per step
    nSamPerStep = 512  #int(math.floor(float(sr) * 0.01))
    # number of steps per frame
    nStepsPerFrame = 3 #int(math.floor(float(nSamPerFrame) / float(nSamPerStep)))
    # real frame size
    nFrameSize = (nStepsPerFrame - 1) * nSamPerStep + n25sSam
    # initial position in the sound stream
    # initPos negative means we need zero padding at the front.
    curPos = nSamPerFrame - nFrameSize
    mfccs = []
    melspecs = []
    chromas = []
    for f in range(0,nFrames):
        # extract features
        frameData, nextPos = extract_one_frame_data(data, curPos, nFrameSize, nSamPerFrame)
        curPos = nextPos
        S = librosa.feature.melspectrogram(frameData, sr, n_mels=128, hop_length=nSamPerStep)
        # 1st is log mel spectrogram
        log_S = librosa.logamplitude(S, ref_power=np.max)
        # 2nd is MFCC and its deltas
        mfcc = librosa.feature.mfcc(y=frameData, sr=sr, hop_length=nSamPerStep, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(delta_mfcc)
        # 3rd is chroma
        chroma = librosa.feature.chroma_cqt(frameData, sr, hop_length=nSamPerStep)        

        full_mfcc = np.concatenate([mfcc[:,0:3].flatten(), delta_mfcc[:,0:3].flatten(), delta2_mfcc[:,0:3].flatten()])
        mfccs.append(full_mfcc.tolist())
        melspecs.append(log_S[:,0:3].flatten().tolist())
        chromas.append(chroma[:,0:3].flatten().tolist())
    return (mfccs, melspecs, chromas)

video_root = "H:/Speech_data/RAVDESS"
audio_root = "H:/Speech_data/RAVDESS_wav"
feat_root = "H:/Speech_data/RAVDESS_feat_new"

def process_all():
    video_dir = plb.Path(video_root)
    feat_dir = plb.Path(feat_root)
    feat_dir.mkdir(parents=True, exist_ok=True)
    for actor in video_dir.iterdir():
        for video_file in actor.iterdir():
            if video_file.name[len(video_file.name)-4:] != '.mp4' or video_file.name[0:2] != '01':
                continue
            seq_dir = plb.Path( feat_root + "/" + actor.name + "/" + video_file.stem )
            if not seq_dir.exists():
                continue
            video_path = str(video_file)
            audio_path = audio_root + "/" + actor.name + "/" + video_file.stem + ".wav"
            mfccs, melspecs, chromas = extract_one_file(video_path, audio_path)
            mfcc_path = feat_root + "/" + actor.name + "/" + video_file.stem + "/mfcc_2.csv"
            mel_path = feat_root + "/" + actor.name + "/" + video_file.stem + "/log_mel.csv"
            chroma_path = feat_root + "/" + actor.name + "/" + video_file.stem + "/chroma_cqt.csv"
            write_csv(mfcc_path, mfccs)
            write_csv(mel_path, melspecs)
            write_csv(chroma_path, chromas)

if __name__ == "__main__":
    process_all()
       
        
    