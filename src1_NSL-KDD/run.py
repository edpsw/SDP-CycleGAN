#!/usr/bin/env python3
from train import Trainer

# Command Line Argument Method
HEIGHT  = 42
WIDTH   = 1
CHANNEL = 1
#LATENT_SPACE_SIZE = 100
EPOCHS = 150
BATCH = 32
CHECKPOINT = 1
#MODEL_TYPE = -1


trainer = Trainer(height=HEIGHT,width=WIDTH, channels=CHANNEL,epochs =EPOCHS,\
                 batch=BATCH,\
                 checkpoint=CHECKPOINT)

# =============================================================================
# trainer = Trainer(height=HEIGHT,\
#                  width=WIDTH,\
#                  channels=CHANNEL,\
#                  #latent_size=LATENT_SPACE_SIZE,\
#                  epochs =EPOCHS,\
#                  batch=BATCH,\
#                  checkpoint=CHECKPOINT,
#                  model_type=MODEL_TYPE)
#                  
# =============================================================================
trainer.train()
