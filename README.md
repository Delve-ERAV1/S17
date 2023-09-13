# S17
# Transformer Models BERT + GPT + ViT

## Overview
Consolidate the codebase for three renowned transformer models: BERT, GPT, and ViT (Vision Transformer). The primary objective is to engineer a unified and streamlined code structure, enabling the training of all three models using just one `transformer.py` file.

## Table of Contents
- [Unified Codebase](#unified-codebase)
- [BERT](#bert)
- [ViT](#vit)
- [GPT](#gpt)

### Unified Codebase
**Objective**: Create a single `transformer.py` file that can be used to train all three models: BERT, GPT, and ViT.

**Steps**:
1. Analyze the existing codebase for each model to identify common components and unique components.
2. Design a modular structure where shared components (like attention mechanisms, feed-forward networks, etc.) are defined once.
3. Implement unique components specific to each model (e.g., masked attention for BERT, causal attention for GPT, etc.).
4. Ensure that the `transformer.py` file has clear interfaces or functions to initiate training for each specific model.

### BERT
```
INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃   ┃ Name      ┃ Type               ┃ Params ┃
┡━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ 0 │ model     │ EncoderTransformer │  7.7 M │
│ 1 │ criterion │ CrossEntropyLoss   │      0 │
└───┴───────────┴────────────────────┴────────┘
Trainable params: 7.7 M                                                                                            
Non-trainable params: 0                                                                                            
Total params: 7.7 M                                                                                                
Total estimated model params size (MB): 30     
```

#### Training Logs
```
Epoch 0/14 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121/121 0:00:46 • 0:00:00 2.57it/s v_num: 1 train_loss_step: 7.757 Δw - 0.3690909090909091
Epoch 1/14 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121/121 0:00:45 • 0:00:00 2.66it/s v_num: 1  train_loss_step: 6.712 train_loss_epoch: 8.647 Δw - 0.1656611570247934
Epoch 2/14 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121/121 0:00:45 • 0:00:00 2.64it/s v_num: 1  train_loss_step: 6.412 train_loss_epoch: 7.086 Δw - 0.18589256198347112
Epoch 3/14 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121/121 0:00:47 • 0:00:00 2.54it/s v_num: 1  train_loss_step: 6.352 train_loss_epoch: 6.483 Δw - 0.36775206611570244
Epoch 4/14 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121/121 0:00:46 • 0:00:00 2.62it/s v_num: 1  train_loss_step: 6.298 train_loss_epoch: 6.351 Δw - 0.7262396694214877
Epoch 5/14 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121/121 0:00:46 • 0:00:00 2.61it/s v_num: 1  train_loss_step: 6.162 train_loss_epoch: 6.305 Δw - 1.121115702479339
Epoch 6/14 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121/121 0:00:46 • 0:00:00 2.61it/s v_num: 1  train_loss_step: 6.179 train_loss_epoch: 6.255 Δw - 1.4677685950413222
Epoch 7/14 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121/121 0:00:45 • 0:00:00 2.65it/s v_num: 1  train_loss_step: 6.167 train_loss_epoch: 6.206 Δw - 1.6524049586776859
Epoch 8/14 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121/121 0:00:46 • 0:00:00 2.64it/s v_num: 1  train_loss_step: 6.106 train_loss_epoch: 6.167 Δw - 1.8169834710743802
Epoch 9/14 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121/121 0:00:46 • 0:00:00 2.58it/s v_num: 1  train_loss_step: 6.124 train_loss_epoch: 6.128 Δw - 1.948099173553719
Epoch 10/14 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121/121 0:00:46 • 0:00:00 2.61it/s v_num: 1  train_loss_step: 6.077 train_loss_epoch: 6.097 Δw - 2.05596694214876
Epoch 11/14 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121/121 0:00:46 • 0:00:00 2.61it/s v_num: 1  train_loss_step: 6.033 train_loss_epoch: 6.049 Δw - 2.197570247933884
Epoch 12/14 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121/121 0:00:46 • 0:00:00 2.63it/s v_num: 1  train_loss_step: 5.971 train_loss_epoch: 6.034 Δw - 2.3406776859504133
Epoch 13/14 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121/121 0:00:45 • 0:00:00 2.64it/s v_num: 1  train_loss_step: 5.898 train_loss_epoch: 6.012 Δw - 2.498768595041322
Epoch 14/14 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121/121 0:00:46 • 0:00:00 2.65it/s v_num: 1  train_loss_step: 5.926 train_loss_epoch: 5.989 Δw - 2.575495867768595
INFO: `Trainer.fit` stopped: `max_epochs=15` reached.
INFO:lightning.pytorch.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=15` reached.
```

### ViT
```
INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃   ┃ Name        ┃ Type               ┃ Params ┃
┡━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ 0 │ accuracy_fn │ MulticlassAccuracy │      0 │
│ 1 │ model       │ EncoderTransformer │ 43.3 M │
│ 2 │ criterion   │ CrossEntropyLoss   │      0 │
└───┴─────────────┴────────────────────┴────────┘
Trainable params: 43.3 M                                                                                           
Non-trainable params: 0                                                                                            
Total params: 43.3 M                                                                                               
Total estimated model params size (MB): 173        
```

#### Training Logs

```
Epoch 0/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8/8 0:00:04 • 0:00:00 2.03it/s v_num: 2 train_loss_step: 0.819 train_acc_step: 0.0                 
Epoch 1/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8/8 0:00:04 • 0:00:00 1.91it/s v_num: 2 train_loss_step: 0.861 train_acc_step: 1.0 val_loss_step:5.249 val_acc_step: 0.0 val_loss_epoch: 3.606 val_acc_epoch: 0.333 train_loss_epoch: 4.29 train_acc_epoch: 0.316              
Epoch 2/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8/8 0:00:04 • 0:00:00 1.87it/s v_num: 2 train_loss_step: 0.756 train_acc_step: 1.0 val_loss_step: 0.011 val_acc_step: 1.0 val_loss_epoch: 3.329 val_acc_epoch:0.413 train_loss_epoch: 2.118 train_acc_epoch: 0.342              
Epoch 3/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8/8 0:00:04 • 0:00:00 1.85it/s v_num: 2 train_loss_step: 1.304 train_acc_step: 0.0 val_loss_step: 2.543 val_acc_step: 0.0 val_loss_epoch: 1.847 val_acc_epoch:0.333 train_loss_epoch: 1.896 train_acc_epoch: 0.338              
Epoch 4/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8/8 0:00:04 • 0:00:00 1.82it/s v_num: 2 train_loss_step: 1.078 train_acc_step: 0.0 val_loss_step: 2.083 val_acc_step: 0.0 val_loss_epoch: 1.508 val_acc_epoch: 0.253 train_loss_epoch: 1.397 train_acc_epoch: 0.307              
Epoch 5/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8/8 0:00:04 • 0:00:00 1.78it/s v_num: 2 train_loss_step: 1.048 train_acc_step: 0.0 val_loss_step:1.307 val_acc_step: 0.0 val_loss_epoch: 1.177 val_acc_epoch:0.333 train_loss_epoch: 1.215 train_acc_epoch: 0.338              
Epoch 6/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8/8 0:00:04 • 0:00:00 1.77it/s v_num: 2 train_loss_step: 1.597 train_acc_step: 0.0 val_loss_step:2.174 val_acc_step: 0.0 val_loss_epoch: 1.42 val_acc_epoch: 0.253 train_loss_epoch: 1.185 train_acc_epoch: 0.307              
Epoch 7/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8/8 0:00:05 • 0:00:00 1.75it/s v_num: 2 train_loss_step: 2.557 train_acc_step: 0.0 val_loss_step:0.83 val_acc_step: 1.0 val_loss_epoch: 1.08 val_acc_epoch: 0.413 train_loss_epoch: 1.199 train_acc_epoch: 0.333              
Epoch 8/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8/8 0:00:04 • 0:00:00 1.80it/s v_num: 2 train_loss_step: 0.894 train_acc_step: 1.0 val_loss_step: 0.801 val_acc_step: 1.0 val_loss_epoch: 1.112 val_acc_epoch:0.413 train_loss_epoch: 1.159 train_acc_epoch: 0.307              
Epoch 9/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8/8 0:00:04 • 0:00:00 1.80it/s v_num: 2 train_loss_step: 1.275 train_acc_step: 0.0 val_loss_step: 1.732 val_acc_step: 0.0 val_loss_epoch: 1.251 val_acc_epoch:0.333 train_loss_epoch: 1.156 train_acc_epoch: 0.356              
Epoch 10/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8/8 0:00:04 • 0:00:00 1.86it/s v_num: 2 train_loss_step: 1.22 train_acc_step: 0.0 val_loss_step: 0.937 val_acc_step: 1.0 val_loss_epoch: 1.081 val_acc_epoch:0.413 train_loss_epoch: 1.206 train_acc_epoch: 0.342         

INFO: `Trainer.fit` stopped: `max_epochs=11` reached.
INFO:lightning.pytorch.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=11` reached.
INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Validation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3/3 0:00:00 • 0:00:00 7.23it/s  
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃      Validate metric      ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│       val_acc_epoch       │    0.41333332657814026    │
│      val_loss_epoch       │     1.082227110862732     │
└───────────────────────────┴───────────────────────────┘
[{'val_loss_epoch': 1.082227110862732, 'val_acc_epoch': 0.41333332657814026}]
```

### GPT
```
INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃   ┃ Name      ┃ Type               ┃ Params ┃
┡━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ 0 │ model     │ DecoderTransformer │ 63.5 M │
│ 1 │ criterion │ CrossEntropyLoss   │      0 │
└───┴───────────┴────────────────────┴────────┘
Trainable params: 63.5 M                                                                                           
Non-trainable params: 0                                                                                            
Total params: 63.5 M                                                                                               
Total estimated model params size (MB): 253  
```

#### Training Logs
- loss metric refers to the training and validation loss
- **iter_loss** refers loss calculated after every train and validation run by sampling the train/validation data **eval_iters** number of times.
```
Epoch 0/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 264/264 0:02:51 • 0:00:00 1.53it/s v_num: 9 train_loss_step: 1.205
Epoch 1/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 264/264 0:02:57 • 0:00:00 1.51it/s v_num: 9  train_loss_step: 0.289 train_loss_epoch: 3.51  train_iter_loss: 1.23             
Epoch 2/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 264/264 0:02:58 • 0:00:00 1.50it/s v_num: 9  train_loss_step: 0.2   train_loss_epoch: 0.554 train_iter_loss: 0.289            
Epoch 3/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 264/264 0:02:57 • 0:00:00 1.51it/s v_num: 9  train_loss_step: 0.174 train_loss_epoch: 0.237 train_iter_loss: 0.2              
Epoch 4/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 264/264 0:02:58 • 0:00:00 1.51it/s v_num: 9  train_loss_step: 0.132 train_loss_epoch: 0.183 train_iter_loss: 0.171   
Epoch 5/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 264/264 0:02:58 • 0:00:00 1.51it/s v_num: 9  train_loss_step: 0.133 train_loss_epoch: 0.16  train_iter_loss: 0.153 val_loss_step: 9.583       val_loss_epoch: 9.637  val_iter_loss: 9.603              
Epoch 6/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 264/264 0:02:58 • 0:00:00 1.51it/s v_num: 9  train_loss_step: 0.147 train_loss_epoch: 0.147 train_iter_loss: 0.142 val_loss_step: 9.583       val_loss_epoch: 9.637  val_iter_loss: 9.603              
Epoch 7/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 264/264 0:02:57 • 0:00:00 1.50it/s v_num: 9  train_loss_step: 0.142  train_loss_epoch: 0.139 train_iter_loss: 0.133 val_loss_step: 9.583      val_loss_epoch: 9.637  val_iter_loss: 9.603              
Epoch 8/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 264/264 0:02:58 • 0:00:00 1.50it/s v_num: 9  train_loss_step: 0.117  train_loss_epoch: 0.131 train_iter_loss: 0.128 val_loss_step: 9.583      val_loss_epoch: 9.637  val_iter_loss: 9.603              
Epoch 9/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 264/264 0:02:59 • 0:00:00 1.43it/s v_num: 9  train_loss_step: 0.12  train_loss_epoch: 0.126  train_iter_loss: 0.126 val_loss_step: 9.583      val_loss_epoch: 9.637 val_iter_loss: 9.603    
```

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃      Validate metric      ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│       val_iter_loss       │    10.185652732849121     │
│      val_loss_epoch       │    10.112564086914062     │
└───────────────────────────┴───────────────────────────┘
[{'val_loss_epoch': 10.112564086914062, 'val_iter_loss': 10.185652732849121}]
```

#### Prediction

* Epoch 0
> [PAD]£ hopeless tunes worthless booker hanson violinvite 315 mysterious pasha bush angular lake beat shuttlesdale 
reaching playingetz bert vector searched sorted inadvertentlyiac अdar之 mastering greece trim kv corner ɐ 
[unused96]ines kingdom mxtation carmel 265 mayfair chronology lear 32 parallelsgic 南 [unused843] schools ahead 
archipelago psychologists stillscks myles multiplied gather verified matthewrner sloop interesting awdiment slalom 
chef bubbling mitochondrialา junction wheelchairpoint per baylor scripts halves journalists domingo zero stuttgart 
encoded pearson newsweek vie 313 ye ponds ached alefires yorkshire hyun baccalaureateを ebert snapped entranceiest

* Epoch 5
> [PAD] that doing there? a linear activation function has two major problems : it is not possible to use 
backpropagation ( gradient descent ) to train the modelthe derivative of the function is a constant and has no 
relation to the input, x. so its not possible to go back and understand which weights in the input neurons can 
provide a better prediction. 2. all layers of the neural network collapse into onewith linear activation functions,
no matter how many layers in the neural network, the last layer will

* Epoch 10
> [PAD] new commit, which is a snapshot of the codebase at a specific point in time. a commit includes a message that
describes the changes that were made. git push : this command is used to push local commits to the git repository 
that you want others to get. git pull : this command is used to pull changes from a remote git repository. this is 
useful when you want to get the latest changes from other developers. git checkout : this command is used to switch

> [PAD] in the ways of model. negative log - likelihood becomes unhappy at smaller values, where it would be limited in capturing " different " weight " kinds of relationships. this is exactly the same patch exactly just that embeddings of each mnist character. if we want to know if a dog is a lab, german shepherd, golden retriever, or poodle, it's somewhere there in the dog of the image's embeddings. if we want to know if the vehicle

> [PAD]. that is distributed more locally. very deep networks are prone to overfitting. it is also hard to pass 
gradient updates through the entire network naively stacking large convolution operations is computationally 
expensive. inception v1 solves this through : with multiple sized filters operating on the same level going wider 
to capture different grf contexts not using addition, but concatenation to make sure grf links are maintained till the softmax layer but this was expensive ( think of the next 3



