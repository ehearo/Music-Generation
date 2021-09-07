import streamlit as st
import os

def app():
    path = os.path.dirname(__file__)
    img_path = path+'/report_image/'
    st.title('Emotional Music Generator')
    link = '[Click here to see the Google Slide](https://docs.google.com/presentation/d/11YxtmQL4nVlYOH0v-_Bfxa8UQTMeGsYiIFX0T8YBmVU/edit?usp=sharing )'
    st.markdown(link, unsafe_allow_html=True)
    st.markdown("""
    ## **Introduction**
    Using both **CNN** for face recognition detects the emotion and  **RNN**  to generate music.          
                """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image(img_path+'001.png')

    st.markdown("""
    ## **Methodology**
    #### Two Stage Model         
                """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image(img_path+'002.png')
    st.markdown("""
    ### **Stage one : Emotion Detector **
    - FER2013 dataset 
    - 28,709 samples in training set
    - 48*48 pixel grayscale image
    - Seven categories
    - Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
                """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image(img_path+'003.png')
    st.markdown("""
    > Facial Emtion Recognizer 
                """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image(img_path+'004.png')
    st.markdown("""
    > Basic Block 
                """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image(img_path+'005.png')
    st.markdown("""
    > FER Training Strategies 
                """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image(img_path+'006.png')
    st.markdown("""
    ### **Stage two:Music Generator **
    - EMOPIA dataset 
    - 1,087 music clips from 387songs
                """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image(img_path+'007.png')
    st.markdown("""
    > Emotion mapping 
                """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image(img_path+'008.png')
    st.markdown("""
    > Music Generator
                """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image(img_path+'009.png')
    st.markdown("""
    ## **Two-way Generate music**  
    ### Midi Files & ABC Notation
                """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image(img_path+'010.png')
    st.markdown("""
    ## **Midi**
    >NoteSeq
     >Get Notes from MIDI using PrettyMIDI
     >One note includes [start, end, pitch, velocity]
                """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image(img_path+'011.png')
    st.markdown("""
    >EventSeq
     >Seprate one noteseq to three eventseq
     >Add a time shift event between two event which have a time gap greater than 0.015
                """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image(img_path+'012.png')
    st.markdown("""
    >EventSeq to an array
                """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image(img_path+'013.png')
    st.markdown("""
    >ControlSeq
     >One event, one control
     >Calculate ‘note on’ histogram and note density 
     >Note_density_bins: [ 1  4  7 10 13 16 19 22 25 28 31 34]
                """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image(img_path+'014.png')
    st.markdown("""
    >ControlSeq to an array
     >[  7  46   0  11   0  11   0   0 173   0   0   0  11]
                """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image(img_path+'015.png')
    st.markdown("""
    ## **ABC Notation**
    >Introduction
                """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image(img_path+'016.png')
    st.markdown("""
    >Data Preprocess
                """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image(img_path+'017.png')
    st.markdown("""
    >Model
                """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image(img_path+'018.png')
    
    st.markdown("""
    ### Result
                """)
#     result_link = '[Click here to see the Result](https://www.abcjs.net/abcjs-editor.html)'
#     st.markdown(result_link, unsafe_allow_html=True)
    
    st.markdown("""
    >NoteSeq"B', '"Dm"fcd "E7"eBd', '"A"A3e aee', '"D"fdf "2f', '"E7"edB "E"A3A ', '"CmA2c e2c', '"A7eec e2c', '"A"e2c edc', '"A"AAd "A7"e', '"CmE2c e2c', '"D7c2c e2c', '"A"e2c eed', '"A7"edB "A"A2:', ' A"EmE3A "2G', '"E"AcA f2d', '"A7cAA "Bc', '"E"BGdB dGB', '"A7cA2eBB', '"Gm"eeB dBd', '"Am"ede "A"A2:', '"A"d2d fef', '"A"f2A Amm"E7" ""Dm"BGB E2A', '"E7"A2e efe', '"D"f2 -2D', '"D"A2A A2f" A"E3FBA "Gm"B2d', '"Em"EBE "2A', '"A7"AAE "2g', '"A"d3 d2:', '"Gm"e2e e2f', '"A7"AGF G2F', '"Gm"G2e e2f', '"Am"EEB "2', '"A7"AEEEE', '"Gm"B2G G2:A', '"A"GBBBG2d', '"C7"ABc AF', '"G"g2G G2', '"C7"Aec eD7"edA', '"G"B3B G2c', '"G7"ABc DE', '"Gm"e2 -E2:" A"E3De "ga', '"C"ggg "2e', '"C7"fef fed', '"G"g2g "D7"g2e', '"G"Be bgg', '"C"ggg "2e', '"C7"fef fed', '"G"g2g "ge', '"C"g2g "ea', '"E""fee def', '"G"g2g "2d', '"Dm"e2d "D""def', '"G"gfg "DEGmb ""C"ggc "Gm"ecg', '"A7"eef "BA', '"G"G3d g2:', '"Cm"e2 -', '"E7AA A2G', '"Cm"E2 -', 'c', '"Am"E2E ece', '"A"d2B GD"A2G', ' ""Em"B3 -:B', '"Cm"AB "2d', '"Am"e2 E2E', '"Em"E3 c2B', '"Am"E'
                """)
    st.markdown("""
    >|"B|"Dm"fcd "E7"eBd|"A"A3e aee|"D"fdf "2f|"E7"edB "E"A3A |"CmA2c e2c|"A7eec e2c|"A"e2c edc|"A"AAd "A7"e|"CmE2c e2c|"D7c2c e2c|"A"e2c eed|"A7"edB "A"A2:| A"EmE3A "2G|"E"AcA f2d|"A7cAA "Bc|"E"BGdB dGB|"A7cA2eBB|"Gm"eeB dBd|"Am"ede "A"A2:|"A"d2d fef|"A"f2A Amm"E7" ""Dm"BGB E2A|"E7"A2e efe|"D"f2 -2D|"D"A2A A2f" A"E3FBA "Gm"B2d|"Em"EBE "2A|"A7"AAE "2g|"A"d3 d2:|"Gm"e2e e2f|"A7"AGF G2F|"Gm"G2e e2f|"Am"EEB "2|"A7"AEEEE|"Gm"B2G G2:A|"A"GBBBG2d|"C7"ABc AF|"G"g2G G2|"C7"Aec eD7"edA|"G"B3B G2c|"G7"ABc DE|"Gm"e2 -E2:" A"E3De "ga|"C"ggg "2e|"C7"fef fed|"G"g2g "D7"g2e|"G"Be bgg|"C"ggg "2e|"C7"fef fed|"G"g2g "ge|"C"g2g "ea|"E""fee def|"G"g2g "2d|"Dm"e2d "D""def|"G"gfg "DEGmb ""C"ggc "Gm"ecg|"A7"eef "BA|"G"G3d g2:|"Cm"e2 -|"E7AA A2G|"Cm"E2 -|c|"Am"E2E ece|"A"d2B GD"A2G| ""Em"B3 -:B|"Cm"AB "2d|"Am"e2 E2E|"Em"E3 c2B|"Am"E|
                """)
    st.markdown("""
    ## Future works and improvements

    ### Extraction of emotions from speech and environment

    Deep learning analysis of mobile physiological, environmental and location sensor data for emotion detection

    ### Music Therapy

    On the use of AI for Generation of Functional Music to Improve Mental Health    


    ## References

    [1] Tan, Xiaodong and M. Antony. “Automated Music Generation for Visual Art through Emotion.” ICCC (2020).

    [2] Madhok, R., Shivali Goel and Shweta Garg. “SentiMozart: Music Generation based on Emotions.” ICAART (2018). 

    [3] Williams, Duncan A. H., Victoria J. Hodge and Chia-Yu Wu. “On the use of AI for Generation of Functional Music to Improve Mental Health.” Frontiers in Artificial Intelligence 3 (2020): n. pag. 

    [4] Wikipedia. 2019. “Music and emotion.” Last modified May 21, 2021. https://en.wikipedia.org/wiki/Music_and_emotion.

    [5] Russell, J.A. (1980). A circumplex model of affect. Journal of Personality and Social Psychology, 39(6), 1161–1178. 

    [6] Karbauskaitė, Rasa, Sakalauskas, Leonidas, and Dzemyda, Gintautas. ‘Kriging Predictor for Facial Emotion Recognition Using Numerical Proximities of Human Emotions’. 1 Jan. 2020 : 249 – 275. 

    [7] Seo, Yeong-Seok & Huh, Jun-Ho. (2019). Automatic Emotion-Based Music Classification for Supporting Intelligent IoT Applications. Electronics. 8. 164. 10.3390/electronics8020164. 

    [8] E. Thayer, Robert (1990). The Biopsychology of Mood and Arousal. Oxford University Press USA.

    [9] Ahn, Junghyun, Stéphane Gobron, Quentin Silvestre and D. Thalmann. “Asymmetrical Facial Expressions based on an Advanced Interpretation of Two-dimensional Russells Emotional Model.” (2010).
    """)