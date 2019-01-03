[//]: # (Image References)

[image1]: images/policy.png "Policy"
[image2]: images/deterministic-policy.png "Deterministic-policy"
[image3]: images/bellman.png "Bellman"
[image4]: images/mseloss.png "Loss"
[image5]: images/policy2.png "Policy"
[image6]: images/algorithm.png "Algorithm"
[image7]: images/noise.png "Noise"
[image8]: images/actor.png "Actor"
[image9]: images/critic.png "Critic"
[image10]: images/training.png "Training"
[image11]: images/test.gif "sample"

#  Continuous Control

### Goal

The goal of the project is to train agent to Solve "Tennis" environment. You can find detailed description of the environment following the [link](README.md) 

### Solution Summary

To solve the environment, we tried MADDPG described in the [paper](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf). However, was not succeeded. So, realizing that even there are two agents they are acting mostly independently and the main goal is to hit the ball over the net - at it actually does not matter a lot if second agent is present on the other side of the net. (There is some cooperation in theory - to keep the ball as long in game as possible agents should hit balls in an easy way for other agent, however in practice racket is too fast and can easily reach any kind of opponent hit).    
So we are going to use Deep deterministic policy gradient (DDPG) algorithm published in the [the Deepmind paper](https://arxiv.org/pdf/1509.02971.pdf) for this environment as well with some small improvements:  
  - using shared experience buffer - so both agent will be contributing their experience in learning proceeds. 
  - using different hyperparameters to make algorithm suitable for this exact environment

The idea of the algorithm is to combine [Policy gradient methods](http://www.scholarpedia.org/article/Policy_gradient_methods) to learn optimal policy together with [DQN algorithm](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) to learn Q function. The problem with base DQN algorithm is that it can solve environments with discrete action space. However, most of the real-world tasks especially in the physical control domain has continues and high dimensional action spaces. DDPG algorithm was designed to solve these issues and learn policies in high-dimensional, continuous action spaces.

We have standard reinforcement learning setup where agent is interacting with environment in discrete timesteps. Environment has high-dimensional continues state space and high-dimensional continuous action spaces. We need to learn optimal policy:

![Policy][image1] 

In our case we are going to learn deterministic policy:

![Policy][image2] 

As usual algorithm is based on the reclusive Bellman equation presented in this way below:

![Bellman][image3] 

Similarly, as for the DQN algorithm, we use function approximator (Neural network) to learn Q function and need to optimize MSE loss below:

![MSE][image4] 

Additionally, we need to use function approximator (Neural network) to approximate optimal policy function:

![Policy][image5]

To find an optimal policy we need our policy function learned in a way to maximize Q value from Bellman equation above. So we use standard method for policy gradient methods - gradient accent. Full algorithm described in a [paper](https://arxiv.org/pdf/1509.02971.pdf) is below:

![Algorithm][image6]

Please note that there are two additional ideas used to improve algorithm convergence

- As mentioned in the paper to favor additional exploration, noise was added to the actions defined by the policy:
![Noise][image7]

- The weights of target networks are softly updated during each steps in contrast to [original DQN paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) where weights of target networks are updated only periodically


### Network archtiecture

The overall architectures we come up with which provide good training results for the Reacher environment are presented on the images below:

Policy network (Actor)

![network architecture][image8]

Q network (Critic) 

![network architecture][image9]

Relu is used as activation function and dropout as regularization technique. Also batch normalization is used for Actor network to imporve convergence.

### Training

Network was trained and environment was solved in 177 episodes. And result can be seen on the diagram below.

![results][image10]

Maximum average score over 100 episodes is 30.594. Which is pretty good result.

PParameters below works well for training.  Changes of most of them do not affect training process in a great way. However, during tons of experiments, I come up with two key components which reduces training time dramatically:
- Adding batch normalization layer to the critic network to the first layer reduced convergence time from 300 – 400 episodes to 170- 200 episodes with all other parameters fixed
- Increase in memory buffer makes training process smoother and less volatile

```python

hyperparams = { "BUFFER_SIZE" : int(1e6),  # replay buffer size
                "BATCH_SIZE" : 1024,        # minibatch size
                "GAMMA" : 0.99,             # discount factor
                "TAU" : 1e-3,               # for soft update of target parameters
                "LR" : 1e-4,                # learning rate 
                "LEARN_EVERY" : 10,         # how often to update the network
                "LEARN_ITERATIONS" : 10,    # how many iterations needed for each network update
              }
              
```

### Next Steps

1. As standard DQN approach is used to learn Q value function in the DDPG it might be considered to try possible DQN imporvements described in the [Rainbow paper](https://arxiv.org/pdf/1710.02298.pdf) to improve training performance.

   - [Dueling DQN](https://arxiv.org/abs/1511.06581)
   - [The Double DQN](https://arxiv.org/abs/1509.06461)
   - [Prioritized experience relay](https://arxiv.org/abs/1511.05952). This one might be especially beneficial in theory to speed up   training as experiences where arm reached a target point in the beginning might be extremely rare and valuable.
   - [Distributional RL](https://arxiv.org/abs/1707.06887)
   - [Noisy nets](https://arxiv.org/abs/1706.10295)


2. Also with slight modification we can apply DDPG algorithm to multiagent environment. The idea there is to have shared experience buffer and single brain for all agents. There might be an improvement in training as agents observe different experience.

3. Finally, we want to implement other Policy gradient methods such as [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) or Actor-Critic methods such as A3C: Asynchronous Advantage Actor-Critic and compare their performance for the "Reacher" environemnt.

### Sample

Here is sample of performance of the DDPG agent with averege score 2.07: ![results][image10]

