"""
PURPOSE: read LibriSpeech data (transcripts and FLAC files) and convert them to prepared pieces of data

METHOD:
    - convert FLAC files to MFCC sequences as numpy ndarrays
    - save MFCC sample with label (string of words) using pickle

RESULT:
    - one pickled sample file per FLAC file
    - sample is a dictionary with keys sample and label
"""

import soundfile as sf
import librosa
import pickle
import os
from settings import *


def get_file_paths(root_dir):
    layer1 = os.listdir(root_dir)
    results = []
    for folder1 in layer1:
        layer2 = os.listdir(root_dir + '/' + folder1)
        for folder2 in layer2:
            layer3 = os.listdir(root_dir + '/' + folder1 + '/' + folder2)
            print(layer3)
            result = {'path': root_dir + '/' + folder1 + '/' + folder2, 'files': layer3}
            results.append(result)
    return results


if __name__ == '__main__':
    data_dirs = get_file_paths(libri_dir)
    for folder in data_dirs:
        samples = []
        labels = []
        for file in folder['files']:
            if 'flac' in file:
                with open(folder['path'] + '/' + file, 'rb') as f:
                    data, samplerate = sf.read(f)
                    sample = librosa.feature.melspectrogram(y=data, sr=samplerate)
                    #sample = librosa.feature.mfcc(y=data, sr=samplerate)
                    samples.append({'data': sample, 'file': file})
            if 'txt' in file:
                with open(folder['path'] + '/' + file, 'r') as f:
                    lines = f.readlines()
                    labels = lines
        #print(samples)
        #print(labels)
        for entry in labels:
            file = entry.split(' ')[0]
            label = entry.replace(file, '').replace('\\n', '').strip()
            for record in samples:
                if file in record['file']:
                    final = {'label': label, 'sample': record['data']}
                    print(file, label)
                    with open(output_dir + '/' + file + '.pickle', 'wb') as outfile:
                        pickle.dump(final, outfile)

