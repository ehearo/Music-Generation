import streamlit as st
import os
import random
import time
import pretty_midi
from midi2audio                            import *
from apps.facial_expression.face_detection import *
from apps.EmotionalMusicGenerator.EmotionalMusicGenerator      import EmotionalMusicGenerator
from apps.EmotionalMusicGenerator.config                       import midi_root, midi_save_dir
from apps.EmotionalMusicGenerator.preprocess                   import MidiProcessor

def app():
    st.title('🔎成果發表')
   

    st.title('Demo音樂')
    selectedViewAudio = st.selectbox("選擇Demo音樂", ["音樂1", "音樂2"])
    
    path = os.path.dirname(__file__)
    my_file = path+'/test_coffin.wav'
    my_file2 = path+'/test_furelise.wav'
    Save_Path = path+'/image/'
    midi_os = path+'/generate_midi/'
    
    if (selectedViewAudio == "音樂1"):
        audio_file = open(my_file, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio / wav')
    elif (selectedViewAudio == '音樂2'):
        audio_file = open(my_file2, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio / wav')
        
    def save_uploadedfile(uploadedfile,num):
        with open(os.path.join(Save_Path,num+uploadedfile.name),"wb") as f:
            f.write(uploadedfile.getbuffer())
        return st.success("Saved File:{} success".format(uploadedfile.name))
    
    def midi_generator(image_dir, midi_dir, len_limit=400):
        st.write("""情緒:""")
        #if not os.path.isfile(midi_save_dir):
           # mp = MidiProcessor()
            #mp.preprocess_midi_files_under(midi_root=midi_root, save_dir=midi_save_dir)
        
        emg = EmotionalMusicGenerator(len_limit)
        status_emotion = emg.generate(image_dir, midi_dir)
        st.write(status_emotion)
        
        return midi_dir
    
    def midi2wav(midi_path):
         midi_file = midi_path
         if midi_file is not None:
             midi_data = pretty_midi.PrettyMIDI(midi_file)
             audio_data = midi_data.fluidsynth()
             audio_data = np.int16(
             audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
             )  # -- Normalize for 16 bit audio https://github.com/jkanner/streamlit-audio/blob/main/helper.py
             virtualfile = io.BytesIO()
             wavfile.write(virtualfile, 44100, audio_data)
             st.audio(virtualfile)
         return True
        
    uploaded_file = st.file_uploader("上傳照片", type=['png','jpeg','jpg'])
    if uploaded_file is not None:
        st.success("上傳成功")
        randnum = str(random.randint(0, 1000)) +'-'
        uploadedFileName = uploaded_file.name
#         midi_path = midi_os + uploaded_file.name +'.MID'
        save_uploadedfile(uploaded_file,randnum)
        with open(uploadedFileName, 'wb') as out:  ## Open temporary file as bytes
            out.write(uploaded_file.read())
        st.image(uploadedFileName)
    
        st.write("""
               # 人臉辨識處理
               """)
        if st.button("辨識"):
            with st.spinner("Loading..."):
                img_path = face_detection(os.path.join(Save_Path, randnum + uploaded_file.name), randnum + uploaded_file.name) 
                st.write("""
                # 情緒辨識及音樂生成
                """)
                if img_path is not None:
                    midi_path = midi_os + os.path.splitext(str(randnum) + uploaded_file.name)[0] +'.MID'
                    gen_midi_dir = midi_generator(img_path, midi_path, 400)
                     if gen_midi_dir is not None:
                         midi2wav(gen_midi_dir)                      
                    
#                    wav_path = os.path.join(path + '/wav_files/', os.path.splitext(randnum + uploaded_file.name)[0] + '.wav')
#                    time.sleep( 2 )
#                    if wav_path is not None:
                        # open wav file

#                        wav_file = open(wav_path, 'rb')
#                        wav_bytes = wav_file.read()
#                        st.audio(wav_bytes, format='audio / wav')
#                    else:
#                        st.write('you uploaded bad file!')