### AUDIO RECORDER  

import os
import streamlit as st
import streamlit.components.v1 as components

import io
import librosa
import numpy as np

import torch
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.pretrained import SpeakerRecognition

import soundfile
import hnswlib
import time
from datetime import datetime


## DESIGN implement changes to the standard streamlit UI/UX
st.set_page_config(page_title="VOICE PASSWORD")
## Design move app further up and remove top padding
st.markdown('''<style>.css-1egvi7u {margin-top: -3rem;}</style>''',
    unsafe_allow_html=True)
## Design change st.Audio to fixed height of 45 pixels
st.markdown('''<style>.stAudio {height: 45px;}</style>''',
    unsafe_allow_html=True)
## Design change hyperlink href link color
st.markdown('''<style>.css-v37k9u a {color: #ff4c4b;}</style>''',
    unsafe_allow_html=True)  # darkmode
st.markdown('''<style>.css-nlntq9 a {color: #ff4c4b;}</style>''',
    unsafe_allow_html=True)  # lightmode



primaryColor = "#919E8B" # green
backgroundColor = "#FBF6F1" # sepia yellow
secondaryBackgroundColor =  "#EBD2B9" # wheat
textColor = "#5D6169" # grey


def save_audio(file):
    if file.size > 4000000:
        return 1
    # if not os.path.exists("audio"):
    #     os.makedirs("audio")
    folder = "audio"
    datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # clear the folder to avoid storage overload
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    try:
        with open("log0.txt", "a") as f:
            f.write(f"{file.name} - {file.size} - {datetoday};\n")
    except:
        pass

    with open(os.path.join(folder, file.name), "wb") as f:
        f.write(file.getbuffer())
    return 0



def audio_to_numpy(filenames):
    x, sr = librosa.load(filenames, sr=30000)
    if x.shape[0] <= 30000:    
        x = np.pad(x, (0, 30000-x.shape[0]), 'constant', constant_values=(0, 0))
        if len(q.shape) == 1:
            x = x[..., None]
    return x       



###CREATING SIDEBAR
# Using object notation
add_selectbox = st.sidebar.selectbox(
    "Please select", 
    ("Home", "Tutorial", "About"), key= 'sidebar')


with st.sidebar:
    st.write('##')
    st.write('##')
    st.write('##')
    st.write('##')    
    st.write('##')    
    st.write('##')    


    rate = st.select_slider(
        'Wanna rate this app?  😎 ',
        options=['awful', 'bad', 'okay', 'good', 'great'])

    if rate == 'awful' or rate == 'bad' or rate =='okay':    
        title = st.text_input('Feedback', '')
        if title != '':
            time.sleep(3)
            st.write('Thank you for your feedback!')

    if rate =='good' or rate=='great':
        txt = st.text_input('Feedback', '')
        if txt != '':
            time.sleep(3)
            st.write('Thank you for your support!')            


if st.session_state.sidebar == 'Home':



    def audiorec_demo_app():

        parent_dir = os.path.dirname(os.path.abspath(__file__))
        # Custom REACT-based component for recording client audio in browser
        build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
        # specify directory and initialize st_audiorec object functionality
        st_audiorec = components.declare_component("st_audiorec", path=build_dir)

        # TITLE and Creator information
        st.title('Voice password')
        st.markdown('Audio recorder implemented by '
            '[Stefan Rummer](https://www.linkedin.com/in/stefanrmmr/) - '
            'view project source code on '
            '[GitHub](https://github.com/stefanrmmr/streamlit_audio_recorder)')
        st.write('\n\n')

        # STREAMLIT AUDIO RECORDER Instance
        st_audiorec()


    if __name__ == '__main__':

        # call main function
        audiorec_demo_app()




        

    # Print the current working directory
    # st.write("Current working directory: {0}".format(os.getcwd()))

    ## Change the current working directory
    #os.chdir('C:/Users/Administrator/Downloads')
    # E:/Finalproject

    # Print the current working directory
    # st.write("New Current working directory: {0}".format(os.getcwd()))



    ### UPLOAD RECORDED AUDIO
   
    uploaded_file = st.file_uploader("Choose a file")
    #st.write("Filename:", uploaded_file.name)

    if uploaded_file is not None:

        ### SPEECH_TO_TEXT
        ## Upload pretrained model

        asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", 
                                                    savedir="pretrained_models/asr-transformer-transformerlm-librispeech",  
                                                    run_opts={"device":"cpu"})
                 
        st.write("#")
               
        if not os.path.exists("audio"):
            os.makedirs("audio")
        path = os.path.join("audio", uploaded_file.name)
        if_save_audio = save_audio(uploaded_file)

        spoken = asr_model.transcribe_file(path)           
        with st.spinner('Processing...'):
             time.sleep(3)
                
        st.write('You said:')
        st.info(spoken)
   
        
    
        ### SPEAKER RECOGNITION

        verifier = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cpu"})


        q = audio_to_numpy(path)
        my_embeddings = np.squeeze(
            verifier.encode_batch(torch.tensor(q)).detach().cpu().numpy())
   
   
         #st.write(my_embeddings.shape)
         #st.write(q.shape)


        ## Upload sample voice

        # Change the current working directory
        #os.chdir('E:/Finalproject')

        #cur = os.getcwd()
        voice_1 = os.path.join('SampleVoice_ggtranslate.wav')


        g = audio_to_numpy(voice_1)
        my_embeddings1 = np.squeeze(
            verifier.encode_batch(torch.tensor(g)).detach().cpu().numpy())
        #st.write(my_embeddings1.shape)
        #st.write(g.shape)



        voice_2 = os.path.join('SampleVoice_kha.wav')

        k = audio_to_numpy(voice_2)
        my_embeddings2 = np.squeeze(
            verifier.encode_batch(torch.tensor(k)).detach().cpu().numpy())
        #st.write(my_embeddings2.shape)
        #st.write(k.shape)






        my_id_1 = 1
        my_id_2 = 2
        # my_id_3 = 3
        # my_id_4 = 4

        p = hnswlib.Index(space = 'cosine', dim = 192)
        p.init_index(max_elements = 1000, ef_construction = 200, M = 16)
        # với my_embedding là embedding voice của các em
        # và my_id là id của các em trong database (ví dụ my_id=0)
        p.add_items(my_embeddings1, my_id_1)
        p.add_items(my_embeddings2, my_id_2)


        # p.add_items(my_embeddings_3, my_id_3)
        # p.add_items(my_embeddings_4, my_id_4)


        # ta thực hiện search bằng dòng code sau
        # vơi labels là array chưa k id giống với target_embed nhất 
        target_embed = my_embeddings

        labels, distances = p.knn_query(target_embed, k = 2)

        #st.write(labels)
        #st.write(distances)


        target_embed = my_embeddings
        labels, distances = p.knn_query(target_embed, k = 2)






        st.write("#")




        if labels[0][0] == 2 and spoken == 'TWO SIX ZERO SIX':
            st.success('Password Correct')
            st.balloons()
            st.snow()
            st.write('Welcome to my Youtube channel. Please click the following link: https://www.youtube.com/channel/UCViAzz3Qtz8IQdUI9DiJ3WA/featured')
        else:
            st.error('Incorrect password or Invalid speaker. Please try again!')


if st.session_state.sidebar == 'Tutorial':

    st.title('Tutorial')

    st.write('This is the `tutorial page` of this application')
    st.write('#')
    # Step1
    st.markdown('##### Step 1: Voice recording')
    st.markdown('- Press `Start Recording` to record your voice password')
    st.markdown('- Click `Stop` to end the audio')
    st.markdown('- If you want to record again, click `Reset` to reset the audio')


    # Step2
    st.markdown('##### Step 2: Audio download')
    st.markdown('- Press `Download` to end the audio')
    st.markdown('- The recorded audio will be downloaded to Downloads Folder on your desktop')

    # Step3
    st.markdown('##### Step 3: Audio upload')
    st.markdown('- Click `Browse files` to upload the audio')
    st.markdown('- Choose your recorded audio in the Downloads Folder')

    # Step4
    st.markdown('##### Step 4: Finish')
    st.markdown('- It will take about 15 sec to process the data')
    st.markdown('- In case of `incorrect password` or `invalid speaker`, click `Χ` next to the uploaded file to delete the audio and record again as from step 1')



if st.session_state.sidebar == 'About':

    st.title('About my project')

    st.markdown('### Project Title: **Application of voice password and speaker verification**')
    st.markdown('#### Project Description')

    st.markdown('''
        - As digital technology advanced in today's world, the potential of privacy violation has been a threat to user's information
        - Thus, this AI application is designed to be capable of verifying user's identity, based on the voice characteristics such as tones, features, and at the same time integrating with voice password authentication.
                 ''')


    st.markdown('#### *Project goals*')
    st.markdown('''
        - Build a security system using voice password authentication combined with speaker recognition as follows:
            - First, with the audio input, the system will verify the voice password before continuing to run the Speaker Recognition Model to identify user. 
            - If both the correct password and target user's voice are matched with the input, the system will navigate the user, or give the user a link to a private website. 
        - The main part this AI model needs to process is to extract features of the speaker's voice to verify it, and to transcribe audio to text.
               ''')


    st.markdown('#### **Scope of work**')
    st.markdown('''    
        - Find an appropriated pretrained model in speech recognition. 
        - Process recorded audio on Streamlit platform. 
        - A completed Streamlit application will be built after accomplishing the basic objectives.
        - After this project, I will be more experienced in data processing related to audio and in deploying an application on Streamlit.
               ''')

    st.markdown('''
        #### *A brief introduction about the project*
        ##### *Abstract*
        - 
        ##### *Model*
        - Speech to text Pretrained Model: [speechbrain_ASR](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-en) 
        - Speaker Verification: [speechbrain_Voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
        ##### *Methods*
        - Applying ASR pretrained model to translate speech to text.
        - Converting audio file into numpy array by librosa module.  
        - Using cosine similarity based on the user's embeddings extracting from the audio to identify voices. 
        ##### *Note*
        - **Reference**:
            - Streamlit audio recorder: https://github.com/stefanrmmr/streamlit_audio_recorder 
            - Streamlit API reference: https://docs.streamlit.io/library/api-reference
        - To set up audio recorder component, read and follow the instruction in [here](https://github.com/stefanrmmr/streamlit_audio_recorder#readme)    
        - Due to being unable to directly return audio recording-data to Python (binary base64), it is required to download the recorded audio. Therefore, change the current working directory to Downloads Folder of your desktop in order to allow the computer to detect to recorded audio file as similar to: ''')
    st.code( "os.chdir('C:/Users/Administrator/Downloads')", language='python')


    st.markdown('''
        - Afterwards, change the working directory back to the directory of your Streamlit project by:
                ''')
    st.code("os.chdir('/home/ _Your_project_folder_')", language='python')


    st.markdown('''
        - To verify speaker, you will need to have at least 2 audio recording from different people, including the target audio that you want the application to recognize. Put those audio in your project folder. and then use the code below to take the path of the audio in your computer. ''')
    sp = '''
              cur = os.getcwd()
voice_1 = os.path.join(cur, '_SampleVoice_audio.wav')
        '''
    st.code(sp, language='python')


        


    st.write('#') 
    st.markdown('''
       #### *Author*
       - Nguyễn Mạnh Kha _ Class of 2022 _ Le Hong Phong High School for the Gifted, Hochiminh City, Vietnam ''')



st.write('#')

st.caption('Made by @khanguyen')  
