### Introduction
End-to-end learning algorithms for tasks like robotic manipulation and autonomous driving often consume raw pixels (in the case of cameras) as a representation of the current environment state. These systems hope to learn a nonlinear mapping from images to control in a way that optimizes some sort of objective. However, it is often hard to successfully train such a visuomotor model to convergence in a way that yields good performance. This is especially obvious in reinforcement learning research, with classic approaches like DQN requiring hundreds of millions of image frames to converge (sometimes unstably). One common hypothesis is that images are a very high-dimensional input, and it may be difficult to efficiently learn a mapping from state to control. For our project, we investigate different methods of visual representation learning for reinforcement learning in the ATARI Pong game. We incorporate auxiliary tasks when training the encoder and decoder to learn visual representations that better inform reinforcement learning policies in these domains. We are able to empirically show that our approach trains DQN policies that are more sample efficient and stable than the baseline DQN approaches. 

- talk about our key result
- bring up RL earlier and tie it to images


### Related Work
Previous work has used multi-task setups to augment the performance of learned planners (ChauffeurNet, Uber NMP). ChauffeurNet is an imitation learning approach that seeks to push the boundaries of completely offline (non-interactive) imitation learning. A major component of their approach is using trajectory perturbations and random object removal to synthesize new and interesting scenarios from recorded driving data. The model is trained to overcome these these synthesized, possibly dangerous scenarios by using a multi-task loss that penalizes bad behavior (e.g. collisions, going off-road). Similarly, the Neural Motion Planner approach uses inverse RL to predict a spatiotemporal cost volume that is used to grade proposed trajectories, with the minimum cost trajectory chosen as the plan. The cost volume is trained alongside auxiliary perception tasks that are claimed to make the learned cost volumes more interpretable.

Our approach similarly uses auxiliary perception tasks for representation learning in reinforcement learning. Our approach is closely related to a recent work called MaRLn that uses end-to-end reinforcement learning for driving in simulation. MaRLn hypothesizes that images are not good representations for RL state because they are too large and contain a lot of irrelevant information. Their idea is to replace the image state with a smaller compressed representation with the key novelty here being how that representation is produced. The authors adopt an autoencoder approach and use a multi-task loss which trains the network to encode the image into a small latent space, and then decode it to perform various tasks like semantic segmentation and some regression tasks. In theory, this latent space should contain information that is relevant to the task at hand while reducing the memory footprint of the replay buffer. 

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

### Technical Approach
#### Training the autoencoder
The training data for the autoencoder is collected by deploying a random agent in Pong, and recording images over a number of episodes. We found that recording 20 episodes worth of data (<10K frames) was sufficient. Formally, our autoencoder $\phi$ trains an encoder $\phi_e$ and decoder $\phi_d$ such that given an image $s$ and ground truth semantic segmentation masks $m$, we minimize $||\phi_d(\phi_e(s)) - m||$. Given image $s$, $m$ can be computed on the fly using color masking - in Pong, the player is green, the opponent is brown, and the ball is white. 

![Pong autoencoder](assets/autoencoder_im1.png)
- examples of segmentation gts

#### Training the DQN
Our baseline is the classic DQN approach which we were able to find a good open-source implementation of [here](https://github.com/Rochan-A/dqn-pong). Deep Q learning is an off-policy RL approach that works by deploying a behavior policy in an environment/task (formulated as an MDP), and filling a *replay buffer* with experience tuples containing (state *s*,action *a*, reward *r*, next state *ns*). The DQN network $Q\_theta$ is trained to optimize **INSERT TD LOSS OBJECTIVE**. During training, the algorithm iterates between (1) rolling out a behavior policy and storing experience tuples in the replay buffer and (2) sampling from the replay buffer and training Q_theta to optimize **TD LOSS**. At test time, we can recover a simple policy $\pi(s) = argmax_aQ(s,a)$. A more detailed explanation of the DQN approach can be found in the [original paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). 

- algorithm
  - DQN - target network

- different models (resnet, custom with different k)
- try ablation with masking out less things

### Results

- how long did it take to get to 21?
- how long did it take to be stable?
  - in wallclock time and steps
- how much space do the encodings take
- ADD IN THE ENCODER TRAINING TIME / STEPS
- what is the best approach tldr
