#!/usr/bin/env python3
from train import Trainer

# Command Line Argument Method
HEIGHT  = 34
WIDTH   = 1
CHANNEL = 1
EPOCHS = 20
BATCH = 128
CHECKPOINT = 1


trainer = Trainer(height=HEIGHT,width=WIDTH, channels=CHANNEL,epochs =EPOCHS,\
                 batch=BATCH,\
                 checkpoint=CHECKPOINT)


trainer.train()
