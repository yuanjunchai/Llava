import sys
import os
# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(base_dir)
# sys.path.append('/mmfs1/gscratch/cse/yjchai/VideoLLM/LLaVA-NeXT')
from llava.train.train import train


if __name__ == "__main__":
    train()
