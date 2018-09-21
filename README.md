# Data Sources

1. LibriSpeech (raw audio and text transcripts)
2. [Universally encoded sentences](https://alpha.tfhub.dev/google/universal-sentence-encoder/2)

# Data Prep

1. Generate MFCC samples from FLAC audio
2. Generate encoded sentences from transcripts

# Datasets Produced

* MFCC samples
* Text sentences
* Encoded sentences

# Encoder Model

Train model on:
* Input: MFCC
* Output: Encoded sentence

# Decoder Model

Train model on:
* Input: Encoded sentence
* Output: Text sentence
