LSTM-IMDB
Try to use lstm to complete sentiment analysis task

## Data
### IMDB
<https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz>

### Pretrained Weights
<http://nlp.stanford.edu/data/glove.6B.zip>


## Train Logs
### scratch 
![scratch loss](./docs/scratch_loss.png)      |  ![scratch acc](./docs/scratch_acc.png)
:-------------------------:|:-------------------------:
Displayed Loss on Tensorboard |  Displayed Accuracy on Tensorboard
```shell script
Epoch 0 : Step 625 => Train Loss: 0.2953 | Train ACC: 0.8750: 100%|██████████| 625/625 [03:58<00:00,  2.62it/s]
Epoch 0 : Step 155 => Val Loss: 0.3365 | Val ACC: 0.8560 
Epoch 1 : Step 625 => Train Loss: 0.1731 | Train ACC: 0.9375: 100%|██████████| 625/625 [03:56<00:00,  2.64it/s]
Epoch 1 : Step 155 => Val Loss: 0.2158 | Val ACC: 0.9123 
Epoch 2 : Step 625 => Train Loss: 0.1563 | Train ACC: 0.9375: 100%|██████████| 625/625 [03:55<00:00,  2.65it/s]
Epoch 2 : Step 155 => Val Loss: 0.1545 | Val ACC: 0.9413 
Epoch 3 : Step 625 => Train Loss: 0.2645 | Train ACC: 0.9062: 100%|██████████| 625/625 [03:56<00:00,  2.64it/s]
Epoch 3 : Step 155 => Val Loss: 0.1377 | Val ACC: 0.9493 
Epoch 4 : Step 625 => Train Loss: 0.1131 | Train ACC: 0.9688: 100%|██████████| 625/625 [03:55<00:00,  2.66it/s]
Epoch 4 : Step 155 => Val Loss: 0.1121 | Val ACC: 0.9593 
```

### use glove model


## TODO