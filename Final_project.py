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
import shutil                                                                                                                                                    



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


def audio_to_numpy(filenames):
    x, sr = librosa.load(filenames, sr=30000)
    if x.shape[0] <= 30000:    
        x = np.pad(x, (0, 30000-x.shape[0]), 'constant', constant_values=(0, 0))
        if len(q.shape) == 1:
            x = x[..., None]
    return x       



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

    ## Rating
    #rate = st.select_slider(
    #    'Wanna rate this app?  😎 ',
    #    options=['awful', 'bad', 'okay', 'good', 'great'])

    #if rate == 'awful' or rate == 'bad' or rate =='okay':    
    #    title = st.text_input('Feedback', '')
    #    if title != '':
    #        time.sleep(3)
    #        st.write('Thank you for your feedback!')

    #if rate =='good' or rate=='great':
    #    txt = st.text_input('Feedback', '')
    #    if txt != '':
    #        time.sleep(3)
    #        st.write('Thank you for your support!')            


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


    
    
    
    ### UPLOAD RECORDED AUDIO
  
    #uploaded_file = st.file_uploader("Choose a file")
    #if uploaded_file is not None:
    
    # new_or_old = st.radio("Are you a new user?",("Yes","No"),key = "radio",horizontal=True)
    #if st.session_state.radio == "No":
    with st.form("my-form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Choose a file")
        submitted = st.form_submit_button("Submit!")

    if submitted and uploaded_file is not None:
        # do stuff with your uploaded file
        if os.path.exists("audio"):
            try:
                shutil.rmtree("audio")
            except OSError as e:
                st.write("Error: %s - %s." % (e.filename, e.strerror))
        
        with st.spinner('Processing...'):
            ### SPEECH_TO_TEXT
            ## Upload pretrained model
            asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", 
                                                            savedir="pretrained_models/asr-transformer-transformerlm-librispeech",  
                                                            run_opts={"device":"cpu"})

            st.write("#")

            if not os.path.exists("audio"):
                os.makedirs("audio")
            path = os.path.join("audio", uploaded_file.name)
            os.listdir("audio")
            
            if_save_audio = save_audio(uploaded_file)

            spoken = asr_model.transcribe_file(path)           
            st.write('You said:')
            st.info(spoken)

            
        ### Speaker Verification   
        verifier = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cpu"})

        ## Upload sample voice
        voice_1 = os.path.join('An.wav')
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

        voice_3 = os.path.join('Tan.wav')
        m = audio_to_numpy(voice_3)
        my_embeddings3 = np.squeeze(
              verifier.encode_batch(torch.tensor(m)).detach().cpu().numpy())

        voice_4 = os.path.join('Phu.wav')
        n = audio_to_numpy(voice_4)
        my_embeddings4 = np.squeeze(
               verifier.encode_batch(torch.tensor(n)).detach().cpu().numpy())


        my_id_1 = 1
        my_id_2 = 2
        my_id_3 = 3
        my_id_4 = 4


        p = hnswlib.Index(space = 'cosine', dim = 192)
        p.init_index(max_elements = 1000, ef_construction = 200, M = 16)
        # my_embedding là embedding voice
        # my_id là id trong database
        p.add_items(my_embeddings1, my_id_1)
        p.add_items(my_embeddings2, my_id_2)
        p.add_items(my_embeddings3, my_id_3)
        p.add_items(my_embeddings4, my_id_4)   

        q = audio_to_numpy(path)
        my_embeddings = np.squeeze(
           verifier.encode_batch(torch.tensor(q)).detach().cpu().numpy())
        #st.write(my_embeddings.shape)
        #st.write(q.shape)
            
        # labels là array chưa k id giống với target_embed nhất 
        target_embed = my_embeddings
        labels, distances = p.knn_query(target_embed, k = 4)

            
        st.write("#")

        if spoken == 'TWO SIX ZERO SIX':  
           st.success('Password Correct')
           if labels[0][0] == 2 and distances[0][0] <0.3:          
               st.balloons()
               st.snow()
               st.write('Welcome to my Youtube channel. Please click the following link: https://www.youtube.com/channel/UCViAzz3Qtz8IQdUI9DiJ3WA/featured')
           else: 
               st.error('Invalid speaker. Please try again!')

        else:
           st.error('Incorrect password. Please try again!')


        with st.sidebar:  

                #st.sidebar.subheader("Voice labels name")
                #col1, col2, col3, col4 = st.columns(4)
                #with col1:
                #    st.markdown("Ân - 1")
                #with col2:
                #    st.markdown("Kha - 2")             
                #with col3:
                #    st.markdown("Tân - 3")                
                #with col4:
                #    st.markdown("Phú - 4")
                #st.write(labels)
                st.write('#')    

                st.sidebar.subheader("Distance to each labels")
                #st.write(distances)

                if labels[0][0]==2:  a="<b>Kha"
                elif labels[0][0]==1:  a="<b>Ân"
                elif labels[0][0]==3:  a="<b>Tân"
                else:    a="<b>Phú"


                if labels[0][1]==2:     b="<b>Kha"
                elif labels[0][1]==1:   b="<b>Ân"
                elif labels[0][1]==3:   b="<b>Tân"
                else:  b="<b>Phú"


                if labels[0][2]==2:  c="<b>Kha"
                elif labels[0][2]==1:  c="<b>Ân"
                elif labels[0][2]==3:  c="<b>Tân"
                else:   c="<b>Phú"


                if labels[0][3]==2: d="<b>Kha" 
                elif labels[0][3]==1: d="<b>Ân" 
                elif labels[0][3]==3: d="<b>Tân" 
                else: d="<b>Phú"


                fig = go.Figure(data=[go.Table(header=dict(values=[['<b>Name'],['<b>Label'],['<b>Distance']],
                                    line_color='darkslategray', fill_color='wheat', align='left', font=dict(color='darkslategray', size=15), height=30),
                                cells=dict(values=[[a,b,c,d],[labels[0][0],labels[0][1],labels[0][2],labels[0][3]], [distances[0][0],distances[0][1],distances[0][2],distances[0][3]]],
                                    line_color='darkslategray', fill_color='wheat', align='left', font=dict(color='darkslategray', size=14), height=30))])
                fig.update_layout(width=290, height=200, margin=dict(l=10, r=0, b=0, t=0))
                st.write(fig)
                st.write('#')    

                st.sidebar.subheader("Recorded audio file")
                file_details = {"Filename": uploaded_file.name, "FileSize": uploaded_file.size}
                st.sidebar.write(file_details)

        #del(uploaded_file)
        #os.remove(path)
        #os.rmdir("audio")
        #try:
        #    shutil.rmtree("audio")
        #except OSError as e:
        #    st.write("Error: %s - %s." % (e.filename, e.strerror))


        #if os.path.exists("audio"):
        #   os.remove(uploaded_file.name)	
                
    if st.button("Clear All"):
        # Clear values from *all* memoized functions:
        st.experimental_memo.clear()            
        st.experimental_singleton.clear()

            
            
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
    st.markdown('- Before uploading a new audio, press `Clear All` to clear the memory to avoid clashes between variables')


if st.session_state.sidebar == 'About':

    st.title('About my project')

    st.markdown('### Project Title: **Application of voice password and speaker verification**')
    st.markdown('#### Project Description')

    st.markdown('''
        - As digital technology advanced in today's world, the potential of privacy violation has been a threat to user's information
        - Thus, this AI application is designed to be capable of verifying user's identity, based on the voice characteristics such as tones, features, and at the same time integrating with voice password authentication.
                 ''')

    st.markdown('''- ######  [GitHub repository of the web-application](https://github.com/Kha1135123/VoiceAuthentication_Finalproject)''')


    st.markdown("##### Theory")
    with st.expander("See Wikipedia definition_Speech Recognition"):
        components.iframe("https://en.wikipedia.org/wiki/Speech_recognition",
                              height=320, scrolling=True)
    with st.expander("See Wikipedia definition_Speaker Recognition"):
        components.iframe("https://en.wikipedia.org/wiki/Speaker_recognition",
                              height=320, scrolling=True)


    st.markdown('#### *Project goals*')
    st.markdown('''
        - Build a security system using voice password authentication combined with speaker recognition as follows:
            - First, with the audio input, the system will verify the voice password before continuing to run the Speaker Recognition Model to identify user. 
            - If both the correct password and target user's voice are matched with the input, the system will navigate the user, or give the user a link to a private website. 
        - The main part this AI model needs to process is to extract features of the speaker's voice to verify it, and to transcribe audio to text.
               ''')


    st.markdown('#### **Scope of work**')
    st.markdown('''    
        - Find an appropriated pretrained model in speech recognition and voice recognition
        - Process recorded audio on Streamlit platform. 
        - A completed Streamlit application will be built after accomplishing the basic objectives.
        - After this project, I will be more experienced in data processing related to audio and in deploying an application on Streamlit.
               ''')

    st.markdown('''
        #### *A brief introduction about the project*
 
        ##### *Model*
        - Speech to text Pretrained Model: [speechbrain/ASR-Wav2Vec2 model -- Commonvoice-en](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-en) 
        - Speaker Verification: [speechbrain/ECAPA-TDNN model -- Voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
        ##### *Methods*
        - Applying ASR pretrained model to translate speech to text.
        - Converting audio file into numpy array by librosa module.  
        - Using cosine similarity based on the user's embeddings extracting from the audio to identify voices by ECAPA-TDNN model. 
        ##### *Note*
        - **Reference**:
            - Streamlit audio recorder: https://github.com/stefanrmmr/streamlit_audio_recorder 
            - Streamlit API reference: https://docs.streamlit.io/library/api-reference
        - To set up audio recorder component, read and follow the instruction in [here](https://github.com/stefanrmmr/streamlit_audio_recorder#readme) ''')   
    st.write("#") 
    st.markdown(''' - If you want to try them we recommend to clone our GitHub repo''')
    st.code("git clone https://github.com/Kha1135123/VoiceAuthentication_Finalproject.git", language='bash')
       
    st.markdown(''' 
    After that, just change the following relevant sections in the Final_project.py file to use this model:
    - Change the current working directory to Downloads Folder of your desktop in order to allow the computer to detect to recorded audio file as similar: ''')
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
