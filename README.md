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

# Inference

1. Generate MFCC from raw audio
2. Generate encoded sentence by feeding MFCC to encoder model
3. Generate text sentence by feeding encoded sentence to decoder model
