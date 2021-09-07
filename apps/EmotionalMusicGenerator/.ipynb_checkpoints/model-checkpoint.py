import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torchvision import transforms
import numpy as np
from apps.EmotionalMusicGenerator.config import device, FER, mapping, midi_save_dir
from PIL import Image
from apps.EmotionalMusicGenerator.utils import find_files_by_extensions
from apps.EmotionalMusicGenerator.sequence import ControlSeq


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class BasicBlock(nn.Module):
    def __init__(self, d_in, stride=1):
        super(BasicBlock, self).__init__()
        self.bb = nn.Sequential(
            nn.Conv2d(d_in, d_in, kernel_size=3, stride=stride, padding=1),
            nn.Conv2d(d_in, d_in, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(d_in, d_in, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(d_in),
            Mish(),
            nn.Conv2d(d_in, d_in, kernel_size=3, stride=stride, padding=1),
            nn.Conv2d(d_in, d_in, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(d_in, d_in, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(d_in),
        )
        self.conv1_1 = nn.Conv2d(d_in * 2, d_in, kernel_size=1, stride=1, padding=0)
        self.r1 = nn.PReLU()

    def forward(self, x):
        identity = x
        x = self.bb(x)
        x = torch.cat((x, identity), dim=1)
        x = self.conv1_1(x)
        x = self.r1(x)
        return x


class FacialEmotionEmbedding(nn.Module):
    def __init__(self, d_in, d_out):
        super(FacialEmotionEmbedding, self).__init__()
        self.conv1 = nn.Conv2d(1, d_in, kernel_size=3, stride=1, padding=1)
        self.bb1 = BasicBlock(d_in, stride=1)
        modules = []
        for i in range(8):
            modules.append(BasicBlock(d_in))
            if i % 3 == 0:
                modules.append(nn.MaxPool2d(kernel_size=2))

        self.sequential = nn.Sequential(*modules)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, d_out),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bb1(x)
        x = self.sequential(x)
        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x


class DeepFaceEmotion(nn.Module):
    def __init__(self, d_out):
        super(DeepFaceEmotion, self).__init__()
        self.Layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2)
        )
        self.Layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2)
        )
        self.Layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, d_out),
        )

    def forward(self, x):
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x


class PerformanceRNN(nn.Module):
    def __init__(
        self,
        event_dim,
        control_dim,
        init_dim,
        hidden_dim,
        gru_layers=3,
        gru_dropout=0.3,
    ):
        super().__init__()
        if 'FacialEmotionEmbedding' in FER['pkl']:
            self.model = FacialEmotionEmbedding(64, 4)
        else:
            self.model = DeepFaceEmotion(4)

        self.model.load_state_dict(torch.load(FER['pkl'],map_location=torch.device('cpu')))
        self.inverse_mapping = {v: k for k, v in mapping.items()}
        self.event_dim = event_dim
        self.control_dim = control_dim
        self.init_dim = init_dim
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.concat_dim = event_dim + 1 + control_dim
        self.input_dim = hidden_dim
        self.output_dim = event_dim

        self.primary_event = self.event_dim - 1
        self.inithid_fc = nn.Linear(init_dim, gru_layers * hidden_dim)
        self.inithid_fc_activation = nn.Tanh()

        self.event_embedding = nn.Embedding(event_dim, event_dim)
        self.concat_input_fc = nn.Linear(self.concat_dim, self.input_dim)
        self.concat_input_fc_activation = nn.LeakyReLU(0.1, inplace=True)

        self.gru = nn.GRU(
            self.input_dim,
            self.hidden_dim,
            num_layers=gru_layers,
            dropout=gru_dropout,
        )
        self.output_fc = nn.Linear(hidden_dim * gru_layers, self.output_dim)
        self.output_fc_activation = nn.Softmax(dim=-1)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.event_embedding.weight)
        nn.init.xavier_normal_(self.inithid_fc.weight)
        self.inithid_fc.bias.data.fill_(0.0)
        nn.init.xavier_normal_(self.concat_input_fc.weight)
        nn.init.xavier_normal_(self.output_fc.weight)
        self.output_fc.bias.data.fill_(0.0)

    def _sample_event(self, output, greedy=True, temperature=1.0):
        if greedy:
            return output.argmax(-1)
        else:
            output = output / temperature
            probs = self.output_fc_activation(output)
            return Categorical(probs).sample()

    def forward(self, event, control=None, hidden=None):
        # One step forward
        assert len(event.shape) == 2
        assert event.shape[0] == 1
        batch_size = event.shape[1]
        event = self.event_embedding(event)

        if control is None:
            default = torch.ones(1, batch_size, 1).to(device)
            control = torch.zeros(1, batch_size, self.control_dim).to(device)
        else:
            default = torch.zeros(1, batch_size, 1).to(device)
            assert control.shape == (1, batch_size, self.control_dim)

        concat = torch.cat([event, default, control], -1)

        input = self.concat_input_fc(concat)
        input = self.concat_input_fc_activation(input)

        _, hidden = self.gru(input, hidden)
        output = hidden.permute(1, 0, 2).contiguous()
        output = output.view(batch_size, -1).unsqueeze(0)
        output = self.output_fc(output)
        return output, hidden

    def get_primary_event(self, batch_size):
        return torch.LongTensor([[self.primary_event] * batch_size]).to(device)

    def init_to_hidden(self, init, images):
        # [batch_size, init_dim]
        batch_size = init.shape[0]
        out, cls = self.load_feature_map(images)
        out = self.inithid_fc(out)
        out = self.inithid_fc_activation(out)
        out = out.view(self.gru_layers, batch_size, self.hidden_dim)
        return out, torch.argmax(cls).data.cpu().numpy()

    def expand_controls(self, controls, steps):
        # [1 or steps, batch_size, control_dim]
        assert len(controls.shape) == 3
        assert controls.shape[2] == self.control_dim
        if controls.shape[0] > 1:
            assert controls.shape[0] >= steps
            return controls[:steps]
        return controls.repeat(steps, 1, 1)

    def generate(
        self,
        init,
        steps,
        events=None,
        controls=None,
        images=None,
        greedy=1.0,
        temperature=1.0,
        teacher_forcing_ratio=1.0,
        output_type="index",
        generate=False,
        generate_control=midi_save_dir
    ):
        # init [batch_size, init_dim]
        # events [steps, batch_size] indeces
        # controls [1 or steps, batch_size, control_dim]
        hidden, image_class = self.init_to_hidden(init, images)
        if generate:
            this_category = self.inverse_mapping[int(image_class)]
            files = list(find_files_by_extensions(generate_control))
            files = [x for x in files if this_category in x]
            assert len(files) > 0, f'no {this_category} file in "{generate_control}"'
            control = np.random.choice(files)
            _, compressed_controls = torch.load(control)
            controls = ControlSeq.recover_compressed_array(compressed_controls)
            controls = torch.tensor(controls, dtype=torch.float32)
            controls = controls.unsqueeze(1).repeat(1, 1, 1).to(device)

        batch_size = init.shape[0]
        assert init.shape[1] == self.init_dim
        assert steps > 0

        use_teacher_forcing = events is not None
        if use_teacher_forcing:
            assert len(events.shape) == 2
            assert events.shape[0] >= steps - 1
            events = events[: steps - 1]

        event = self.get_primary_event(batch_size)
        use_control = controls is not None
        if use_control and not generate:
            controls = self.expand_controls(controls, steps)

        outputs = []
        step_iter = range(steps)

        for step in step_iter:
            try:
                control = controls[step].unsqueeze(0) if use_control else None
            except:
                control = None
            output, hidden = self.forward(event, control, hidden)

            use_greedy = np.random.random() < greedy
            event = self._sample_event(
                output, greedy=use_greedy, temperature=temperature
            )

            if output_type == "index":
                outputs.append(event)
            elif output_type == "softmax":
                outputs.append(self.output_fc_activation(output))
            elif output_type == "logit":
                outputs.append(output)
            else:
                assert False

            if use_teacher_forcing and step < steps - 1:  # avoid last one
                if np.random.random() <= teacher_forcing_ratio:
                    event = events[step].unsqueeze(0)

        return torch.cat(outputs, 0), int(image_class)

    def load_feature_map(self, images):
        feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
        feature_extractor.eval()

        class_extractor = self.model
        class_extractor.eval()

        preprocess = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(50),
                transforms.CenterCrop(48),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485], std=[0.229]
                ),
            ]
        )

        input_batch = []

        for image_path in images:
            input_image = Image.open(image_path)
            input_tensor = preprocess(input_image)
            input_batch.append(input_tensor)

        input_batch = torch.stack(input_batch)
        # input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
        if torch.cuda.is_available():
            input_batch = input_batch.to("cuda")
            feature_extractor.to("cuda")
            class_extractor.to("cuda")

        with torch.no_grad():
            output = feature_extractor(input_batch)
            output_class = class_extractor(input_batch)

        feature_map = output[..., 0]
        return feature_map.transpose(1, 2), output_class
