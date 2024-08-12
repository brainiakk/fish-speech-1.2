import os

import pygame

from modules.fish_speech.tools.llama.generate import main as generate
from pathlib import Path
from modules.fish_speech.tools.vqgan.inference import main as infer

class VoiceService:
    def __init__(self):
        self._output_dir = "outputs/"
        os.makedirs(self._output_dir, exist_ok=True)

    def fishspeech(self, text):
        infer(input_path=Path("austin.wav"), output_path=Path(self._output_dir+"fake.wav"),
              checkpoint_path="checkpoints/fish-speech-1.2-sft/firefly-gan-vq-fsq-4x1024-42hz-generator.pth",
              config_name="firefly_gan_vq", device="cpu")

        generate(text=text,
                 prompt_text=[
                     # "accessing alarm and interface settings in this window you can set up your customized greeting and alarm preferences the world needs your expertise or at least your presence launching a series of displays to help guide you",
                     "Please find attached the comprehensive list of issues we noticed while reviewing the apps. Unfortunately, some of them are connected and do not help us move to the next level to find out other issues therein. This means it is possible that when some of these issues are resolved we would still find other issues that are of concern. If you need clarification on any specific issue please do not hesitate to let me know."],
                 prompt_tokens=[Path(self._output_dir+"fake.npy")],
                 checkpoint_path=Path("checkpoints/fish-speech-1.2-sft"),
                 half=True,
                 device="cpu",
                 num_samples=3,
                 max_new_tokens=0,
                 top_p=0.7,
                 repetition_penalty=1.2,
                 temperature=0.3,
                 compile=False,
                 seed=42,
                 iterative_prompt=True,
                 chunk_length=200)


        infer(input_path=Path("codes_2.npy"), output_path=Path(self._output_dir+"output.wav"),
              checkpoint_path="checkpoints/fish-speech-1.2-sft/firefly-gan-vq-fsq-4x1024-42hz-generator.pth",
              config_name="firefly_gan_vq", device="cpu")

        self.play(self._output_dir+"output.wav")

    def play(self, temp_audio_file):
        pygame.mixer.quit()
        pygame.mixer.init()
        pygame.mixer.music.load(temp_audio_file)
        pygame.mixer.music.stop()
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.music.stop()
        pygame.mixer.quit()

        # os.remove(temp_audio_file)
