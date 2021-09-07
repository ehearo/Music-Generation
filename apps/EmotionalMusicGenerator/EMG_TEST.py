import os
from EmotionalMusicGenerator import EmotionalMusicGenerator
from config import midi_root, midi_save_dir
from preprocess import MidiProcessor

if not os.path.isfile(midi_save_dir):
    mp = MidiProcessor()
    mp.preprocess_midi_files_under(midi_root=midi_root, save_dir=midi_save_dir)
    

# 400是給定midi生成最大長度
# 可以自己調整一下
emg = EmotionalMusicGenerator(400)

# 第一個參數是輸入影像位置
# 第二個參數是輸出MID位置
this_emotion = emg.generate(
    '../dataset/emotion_image/Q1/Q1_Training_10028230.jpg',
    'output/Q1_Training_44083.MID'
)
print(this_emotion)

    
    