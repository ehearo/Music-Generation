import torch
from apps.EmotionalMusicGenerator.model import PerformanceRNN
from apps.EmotionalMusicGenerator.config import device
from apps.EmotionalMusicGenerator.utils import event_indeces_to_midi_file
import os


class EmotionalMusicGenerator:
    def __init__(self, max_len):
        """
        MIDI產生器
        :param max_len: MID長度
        """
        path = os.path.dirname(__file__)
        model_path = path+'/models/PerformanceRNN.sess'
        
        self.init_zero = False
        self.sess_path = model_path
        self.max_len = max_len
        self.greedy_ratio = 1
        self.temperature = 1
        self.state = torch.load(self.sess_path,map_location=torch.device('cpu'))
        self.model = PerformanceRNN(**self.state["model_config"]).to(device)
        self.model.load_state_dict(self.state["model_state"])
        self.model.eval()
        self.mapping = {
            0: 'Positive',
            1: 'Negative',
            2: 'Sadness',
            3: 'Neutral'
        }

    def generate(self, input_file, output_file):
        """
        音樂生成
        :param input_file: 輸入影像絕對位置
        :param output_file: 輸出音檔絕對位置
        :return:
        """
        if self.init_zero:
            init = torch.zeros(1, self.model.init_dim).to(device)
        else:
            init = torch.randn(1, self.model.init_dim).to(device)

        with torch.no_grad():
            outputs, emotion = self.model.generate(
                init,
                self.max_len,
                images=[input_file],
                greedy=self.greedy_ratio,
                temperature=self.temperature,
                generate=True,
            )
            output = outputs.cpu().numpy().T[0]
            torch.cuda.empty_cache()
            event_indeces_to_midi_file(output, output_file)
        return self.mapping[emotion]


if __name__ == '__main__':
    emg = EmotionalMusicGenerator(400)
    this_emotion = emg.generate(
        'C:/dataset/emotion_image/Q3/Q3_Training_24001808.jpg',
        'output/Q3_Training_24001808.MID'
    )
    print(this_emotion)



