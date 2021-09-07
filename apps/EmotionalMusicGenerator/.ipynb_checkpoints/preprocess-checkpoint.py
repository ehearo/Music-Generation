import os
import torch
import hashlib
from random import random
from apps.EmotionalMusicGenerator.sequence import NoteSeq, EventSeq, ControlSeq
from apps.EmotionalMusicGenerator.config import midi_root, midi_save_dir
from apps.EmotionalMusicGenerator.utils import find_files_by_extensions


class MidiProcessor:
    def __init__(self):
        self.readme = 'hi'

    @staticmethod
    def preprocess_midi(path):
        note_seq = NoteSeq.from_midi_file(path)
        note_seq.adjust_time(-note_seq.notes[0].start)
        event_seq = EventSeq.from_note_seq(note_seq)
        control_seq = ControlSeq.from_event_seq(event_seq)
        return event_seq.to_array(), control_seq.to_compressed_array()

    def preprocess_midi_files_under(self, midi_root, save_dir):

        midi_paths = list(
            find_files_by_extensions(midi_root, [".mid", ".midi"])
        )
        os.makedirs(save_dir, exist_ok=True)
        out_fmt = "{}-{}.data"

        for path in midi_paths:
            if random() < 0.95:
                continue
#             print(" ", end="[{}]".format(path), flush=True)
            try:
                data = self.preprocess_midi(path)
            except KeyboardInterrupt:
                print(" Abort")
                return
            except:
                print(" Error")
                continue

            name = os.path.basename(path)
            code = hashlib.md5(path.encode()).hexdigest()
            save_path = os.path.join(save_dir, out_fmt.format(name, code))
            torch.save(data, save_path)


if __name__ == "__main__":
    mp = MidiProcessor()
    mp.preprocess_midi_files_under(midi_root=midi_root, save_dir=midi_save_dir)

