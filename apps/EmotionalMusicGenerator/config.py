import torch
from apps.EmotionalMusicGenerator.sequence import EventSeq, ControlSeq

# MidiProcessor
#midi_root = '/project/at101-group15/dataset/midis'
#midi_save_dir = '/project/at101-group15/dataset/midi/rnn'

# data dir
#image_dir = '/project/at101-group15/dataset/emotion_image'

# Category
mapping = {
    'Q1': 0,
    'Q2': 1,
    'Q3': 2,
    'Q4': 3,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fer_config = {
    'image_size': 48
}

FER = {
    'pkl': '/apps/EmotionalMusicGenerator/models/FacialEmotionEmbedding.pkl'
}

model = {
    'init_dim': 64,
    'event_dim': EventSeq.dim(),
    'control_dim': ControlSeq.dim(),
    'hidden_dim': 512,
    'gru_layers': 4,
    'gru_dropout': 0.3,
}

train = {
    'learning_rate': 0.001,
    'batch_size': 16,
    'window_size': 300,
    'stride_size': 20,
    'use_transposition': True,
    'control_ratio': 1,
    'teacher_forcing_ratio': 1
}
