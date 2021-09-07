from midi2audio import FluidSynth
from shutil import copyfile
import os
import time

fs = FluidSynth()

midi_dir = 'generate_midi/'
bak_dir = 'generate_midi_bak/'
wav_dif = 'wav_files/'


while True:
    midi_list = [x for x in os.listdir(midi_dir) if 'ipynb_checkpoints' not in x]
    if len(midi_list) > 0:
        for midi in midi_list:
            st = time.time()
#             name = midi.split('.')[: -1]
            name = os.path.splitext(midi)[0]
#             wav_name = '.'.join(name) + '.wav'
            wav_name = name + '.wav'
            try:
                fs.midi_to_audio(midi_dir + midi, wav_dif + wav_name)
                print('cost:', time.time() - st)
                copyfile(midi_dir + midi, bak_dir + midi)
                os.remove(midi_dir + midi)
            except Exception as e:
                print(e)