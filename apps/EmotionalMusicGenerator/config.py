import torch
from apps.EmotionalMusicGenerator.sequence import EventSeq, ControlSeq

# MidiProcessor
midi_root = '/app/emotion-music-gan/apps/EmotionalMusicGenerator/dataset/midis'
midi_save_dir = '/app/emotion-music-gan/apps/EmotionalMusicGenerator/dataset/midi_save_dir'

# data dir
image_dir = '/app/emotion-music-gan/apps/EmotionalMusicGenerator/dataset/emotion_image'

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
    'pkl': '/app/emotion-music-gan/apps/EmotionalMusicGenerator/models/FacialEmotionEmbedding.pkl'
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
