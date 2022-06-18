---
title: Voice password app
emoji: 🤗
colorFrom: yellow
colorTo: orange
sdk: streamlit
app_file: Final_project.py
pinned: false
---

# VoiceAuthentication_Finalproject

## Abstract

- This project is an application of voice password and speaker verification training via 2 pretrained model: [ASR_wav2vec2](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-en) and [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- Module used in this project: 
	- [Speechbrain](https://speechbrain.github.io/)
	- [Torch](https://pytorch.org/)
	- [Librosa](https://librosa.org/)
	- [Numpy](https://numpy.org/)
- Library used in this project:
    - [Streamlit](https://streamlit.io/)
    - [Hnswlib](https://github.com/nmslib/hnswlib)
    - [Plotly](https://plotly.com/python/)


## Introduction
- As digital technology advanced in today's world, the potential of privacy violation has been a threat to user's information
- Thus, this AI application is designed to be capable of verifying user's identity, based on the voice characteristics such as tones, features, and at the same time integrating with voice password authentication
- The overall methods are to apply an Automatic Speech Recognition (ASR) pretrained model to translate speech to text, and using cosine similarity based on the user's embeddings extracted from the audio via ECAPA-TDNN model to identify voices. 

## Content
### Data used in pretrained model
- Speech Recognition: [CommonVoice_En](https://commonvoice.mozilla.org/en/datasets)
- Each entry in the dataset consists of a unique MP3 and corresponding text file. Many of the 20,217 recorded hours in the dataset also include demographic metadata like age, sex, and accent that can help train the accuracy of speech recognition engines

```
Size: 71 GB
Number of voices: 81,085
Audio format: MP3
```
<br>

- Speaker Verification: [Voxceleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) [Voxceleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)

```
Speaker: 7,000 +
Utterances: 1 millions +
Hours of audio and video: 2,000 +
```

### Code and Presentation
- [Presentation slide](https://hackmd.io/@Kha/ry3VnpVK9#/)
- Codes: 
	- [Speech Recognition](https://github.com/Kha1135123/VoiceAuthentication_Finalproject/blob/master/Final_project.py): transcribe speech to text
	- [Speaker Verification](https://github.com/Kha1135123/VoiceAuthentication_Finalproject/blob/master/Final_project.py): Calculate the nearest cosine distance of the speaker's embedding vector to the target vector
	- [Streamlit app](https://share.streamlit.io/kha1135123/voiceauthentication_finalproject/Final_project.py): A website of the application
	
### Note: 
- **Reference**:

     - Streamlit audio recorder: https://github.com/stefanrmmr/streamlit_audio_recorder 
     - Streamlit API reference: https://docs.streamlit.io/library/api-reference
     - To set up audio recorder component, read and follow the instruction in [here](https://github.com/stefanrmmr/streamlit_audio_recorder#readme)  
- If you want to try them we recommend to clone our GitHub repo
```
$ git clone https://github.com/Kha1135123/VoiceAuthentication_Finalproject
```       
- After that, modify the following relevant sections in the Final_project.py file to use this model:

    - Change the current working directory to Downloads Folder of your desktop in order to allow the computer to detect to downloaded audio file as similar:
    ```
    os.chdir('C:/Users/Administrator/Downloads'
    ```
    - Afterwards, change the working directory back to the home directory:
    ```
    os.chdir('/home/ _Your_project_folder_')
    ```    
    - To verify speaker, you will need to have at least 2 audio recording from different people, including the target audio that you want the application to recognize. Put those audio in your project folder. and then use the code below to take the path of the audio in your computer.
    ```
    cur = os.getcwd()
    voice_1 = os.path.join(cur, '_SampleVoice_audio.wav')
    ```
### Contributor
- Nguyen Manh Kha - Class of 2022 - Le Hong Phong High School for the Gifted,
  Ho Chi Minh City, Vietnam.

