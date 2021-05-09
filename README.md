### Introduction
End-to-end learning algorithms for tasks like robotic manipulation and autonomous driving often consume raw pixels (in the case of cameras) as a representation of the current environment state. These systems hope to learn a nonlinear mapping from images to control in a way that optimizes some sort of objective. However, it is often hard to successfully train such a visuomotor model to convergence in a way that yields good performance, especially for complex tasks like autonomous navigation. One common hypothesis is that images are a very high-dimensional input, and it may be difficult to efficiently learn a mapping from state to control. For our project, we investigate different methods of visual representation learning for planning and action in video games. We incorporate auxiliary tasks when training the encoder and decoder to learn visual representations that better inform reinforcement learning policies in these domains.

- talk about our key result
- bring up RL earlier and tie it to images


### Related Work
Previous work has used multi-task setups to augment the performance of learned planners (ChauffeurNet, Uber NMP). One recent work has specifically tried to learn "implicit affordances" - a compact learned representation - for reinforcement learning in autonomous driving. 

- Note - we want to focus on how awful it is to have high dimensional state
- Need to talk about DQN original results and the drawbacks

### Approach
The classic ATARI games are a simple testbed for training and evaluating autonomous agents and are popular because they naturally provide interesting, dynamic environments (often with an explicit reward structure). We worked with the classic Pong game. The OpenAI Pong implementation has three main classes of actors: the player agent, the other agent, and the ball. The goal of the game is to get the ball past the other agent. 

- RL reward structure

Our key idea is to use an autoencoder to compress the representation of the state from a high-dimensional image input (the game screen) to a smaller latent space. 

- explain how using the prediction task will force the encoder to learn something useful intuition
- "positional knowledge"
- you need to know where other things are to play this game
- there are non-useful things
- if you know what things are important you can force the encoder to ignore the useless stuff

Each channel of the decoder predicts the pixel locations of a particular class of object. This is trained with ground truth segmentations and forces the encoded latent space to maintain information about task-relevant objects in the game. The specifics of the auxiliary tasks will likely differ depending on the testing environment. After training the encoder, we can try to train reinforcement/imitation learning approaches using encoded images instead of raw images, and compare performance on the appropriate benchmarks. 

Our plan is to use the encoder-decoder setup mentioned previously where we ask the decoder to reconstruct the locations of each class of agent as the auxiliary task. For example, we could produce a segmentation mask of the bullets in the scene, and weight this reconstruction loss very high (since bullets are especially relevant to staying alive). There's no straightforward API for retrieving semantically segmented masks of the game window, but we can use basic heuristics to retrieve the masks (e.g. space invaders are color-coded by class). 

#### Technical Approach

- algorithm
  - DQN - target network
- how did we do segmentation
- examples of segmentation gts
- different models (resnet, custom with different k)
- try ablation with masking out less things

### Results

- how long did it take to get to 21?
- how long did it take to be stable?
  - in wallclock time and steps
- how much space do the encodings take
- ADD IN THE ENCODER TRAINING TIME / STEPS
- what is the best approach tldr