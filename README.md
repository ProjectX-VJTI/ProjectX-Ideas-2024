# ProjectX-Ideas-2024
This is the tentative list of project ideas for the 2024 programme.

Relevant information about each of the projects are provided, and those interested can contact the mentors in case of any query.

Note that this list is subject to change and not complete.

***
## Virtual Try On for Products
### Description

Image visual try-on aims at transferring a target clothes image onto a reference person, and has become a hot topic in recent years. Prior works usually focus on preserving the character of a clothes image (e.g. texture, logo, embroidery) when warping it to arbitrary human pose. However, it remains a big challenge to generate photo-realistic try-on images when large occlusions and human poses are presented in the reference person. 

This is a topic which will include Augmented Reality(AR) combined with Machine Learning(ML) to detect body movements. There are many ways to tackle this issue. One of them being:

Step 1: A semantic layout generation module utilizes semantic segmentation of the reference image to progressively predict the desired semantic layout after try-on.

Step 2: A clothes warping module warps clothes image according to the generated semantic layout, where a second-order difference constraint is introduced to stabilize the warping process during training. 

Step 3: An inpainting module for content fusion integrates all information (e.g. reference image, semantic layout, warped clothes) to adaptively produce each semantic part of human body. 

This is one of the ways in which this problem can be tackled but not the only one. Applicants are encouraged to come up with your own solutions and steps to complete this project which could be more efficient and unique. 

### References
[1] [Human Localization in Real-Time Video](https://drive.google.com/file/d/1AcBzRNdpn2sMVZ81QY057rZrboTo2ZMy/view?usp=sharing
)

[2] [Virtual Try-On Implementation](https://drive.google.com/file/d/13_5T6FjZ-lgoM12CvfxSDrWREuEAgN53/view?usp=sharing
)

### Examples
![Image1](https://drive.google.com/uc?id=1fgF3xOsGL9bmAqsCVIecRaYBXr_faOV6)

![Image2](https://drive.google.com/uc?id=1A7AFluZUvBHL1dVuWjiFuiTOyp7VCWde)

**Difficulty:** Hard

**Mentors:** Mrudul Pawar

**Domains:** Computer Vision, Deep Learning, Augmented Reality

***

## GPT Reimagined: KANs vs MLPs
### Description
In this project we will be implementing Kolmogorov-Arnold Networks (KANs) which are a great alternative for traditional Multi Layer Perceptrons (MLPs). KANs have been invented very recently and are found to outperform the traditional MLP approach. We will implement GPTs (Generative Pretrained Transformers) using the traditional MLP approach and later using KANs to give a proof of principle for the performances of each of them.

### References
[1] [MLP vs KAN](https://www.youtube.com/watch?v=-PFIkkwWdnM)

[2] [KANs explained](https://www.youtube.com/watch?v=7zpz_AlFW2w)

[3] [Original Research Paper proposing KANs](https://arxiv.org/pdf/2404.19756)

**Difficulty:** Hard

**Mentors:** Param Thakkar, Mayank Palan

**Domains:** Deep Learning

***

## Super MaRLo Bros
### Description
![Super Mario Bros](https://i.insider.com/560ebbe7dd0895325c8b458e?width=500)

Inspired by the classic game Super Mario Bros, Super MaRLo Bros is an exciting project that combines game development and artificial intelligence. You'll build the game from scratch using PyGame and then train the computer to navigate and succeed in it using Reinforcement Learning. You have to teach the computer to play the game you made. We'll experiment with different RL algorithms, learning how they work and which ones are most effective as we go along the project. This project is perfect for anyone interested in game development and artificial intelligence.

Reinforcement Learning (RL) is a machine learning technique that trains software to make optimal decisions through trial and error. RL algorithms use a reward-and-punishment system to process data. Actions that help achieve the goal are reinforced, while others are ignored.

By the end, you'll have a fully functional game and an AI-based smart computer player! No need to get overwhelmed, it only needs your interest!

### References
[1] [PyGame](https://medium.com/iothincvit/pygame-for-beginners-234da7d3c56f)

[2] [Reinforcement Learning](http://arxiv.org/pdf/1909.04751)

[3] [Basics of RL](https://www.slideshare.net/slideshow/a-brief-overview-of-reinforcement-learning-applied-to-games/110276440)

[4] [Fundamental of Mario](https://drive.google.com/file/d/1Q8-6vndo1GYvX_cduDS_AlgZMj5bbJMP/view?usp=sharing)

**Difficulty:** Medium to Hard

**Mentors:** Vedant Mehra, Labhansh Naik

***

