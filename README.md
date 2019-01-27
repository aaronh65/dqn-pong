# Reinforcement Learning for Atari Pong using DQN

> NOTE:
This is an unofficial fork of the original code published as [dqn-mario](https://github.com/nailo2c/dqn-mario).

# DQN

使用PyTorch實作DQN演算法，以及atari-pong，整體架構參考openai/baselines。

*Warning*：訓練DQN請開足夠的記憶體，Replay Buffer以預設值1000000為例至少會使用約8G的記憶體.
# Dependencies

* Python 3.6
* PyTorch
* gym
* gym[atari]

# References

[Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)  
[openai/baselines](https://github.com/openai/baselines)  
[transedward/pytorch-dqn](https://github.com/transedward/pytorch-dqn)  
[openai/gym](https://github.com/openai/gym)  