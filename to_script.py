import torch
import requests
import datetime
import json
import time

import numpy as np
import torch
from tqdm import tqdm
from tsai.all import *

model=InceptionTimeXLPlus(1,2)
model.load_state_dict(torch.load('save/icptxlp_e60_2cls.pt', map_location='cpu'))
model.eval()
smod=torch.jit.script(model)
smod.eval()
smod.save("scrited_model.pt1")
