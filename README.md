# ProjectX-Ideas-2024
This is the tentative list of project ideas for the 2024 programme.

Relevant information about each of the projects are provided, and those interested can contact the mentors in case of any query.

Pre-requisites mentioned are not enforced in any way, and are just there to give you a sense of what skills are needed for the project, although it is a plus point if you have the pre-requisites :)

Note that this list is subject to change and not complete.

***
## Virtual Try On for Products
### Description

Image visual try-on aims at transferring a target clothes image onto a reference person, and has become a hot topic in recent years. Prior works usually focus on preserving the character of a clothes image (e.g. texture, logo, embroidery) when warping it to arbitrary human pose. However, it remains a big challenge to generate photo-realistic try-on images when large occlusions and human poses are presented in the reference person. 

This is a topic which will include Augmented Reality(AR) combined with Machine Learning(ML) to detect body movements. There are many ways to tackle this issue. One of them being:

1. A semantic layout generation module utilizes semantic segmentation of the reference image to progressively predict the desired semantic layout after try-on.

2. A clothes warping module warps clothes image according to the generated semantic layout, where a second-order difference constraint is introduced to stabilize the warping process during training. 

3. An inpainting module for content fusion integrates all information (e.g. reference image, semantic layout, warped clothes) to adaptively produce each semantic part of human body. 

This is one of the ways in which this problem can be tackled but not the only one. Applicants are encouraged to come up with your own solutions and steps to complete this project which could be more efficient and unique. 

### References
[1] [Human Localization in Real-Time Video](https://drive.google.com/file/d/1AcBzRNdpn2sMVZ81QY057rZrboTo2ZMy/view?usp=sharing
)

[2] [Virtual Try-On Implementation](https://drive.google.com/file/d/13_5T6FjZ-lgoM12CvfxSDrWREuEAgN53/view?usp=sharing
)

### Examples
![Image1](https://drive.google.com/uc?id=1fgF3xOsGL9bmAqsCVIecRaYBXr_faOV6)

![Image2](https://drive.google.com/uc?id=1A7AFluZUvBHL1dVuWjiFuiTOyp7VCWde)

**Pre-requisites:** C++ Programming, Python Programming, Basic Understanding of Computer Vision

**Difficulty:** Hard

**Mentors:** Mrudul Pawar

**Domains:** Computer Vision, Deep Learning, Augmented Reality

***
## Text To Speech
### Description

![Text To Speech](https://videocdn.geeksforgeeks.org/geeksforgeeks/ConvertTexttoSpeechinPython/ConvertTexttoSpeechUsingPython20221025115334.jpg)

![Stephen Hawking](https://images.theconversation.com/files/210717/original/file-20180316-104645-6iiy85.jpg?ixlib=rb-4.1.0&rect=0%2C97%2C1620%2C810&q=45&auto=format&w=1356&h=668&fit=crop)

Text To Speech Synthesis is a machine learning task that involves converting written text into spoken words. The goal is to generate synthetic speech that sounds natural and resembles human speech as closely as possible.

Primitively, this was done by storing recorded clips of a person making various sounds like "ri-" or "-zz" and construct speech from this database by mapping sequences of alphabets to these sounds. However, with the rise of Deep Learning, many more efficient methods have emerged that perform much better at this task.

Normally, Deep Learning Models dealing with TTS comprise of a frontend and a backend, the frontend converts character sequences to spectrograms, and the backend(vocoder) converts spectrograms to audio. The frontend and backend used however differ widely across solutions.

This project will be exploring a wide number of these methods and implementing the ones that work best. Text To Speech has **huge** applications, some of which include, helping the blind browse websites, giving a voice to the mute(remember Stephen Hawking?), making announcements in public places(airports, train stations) and so, so much more.

Additionally, more research is ongoing to see how feasible it is to create a TTS model that mimics someone else voice(deepfakes) given audio clips of that person speaking.

The sky is the limit with this project, all you need is the hardwork and commitment to see it through.

### References

[1] [Understanding Text To Speech](https://en.wikipedia.org/wiki/Speech_synthesis)

[2] [Understanding Audio Processing and Spectrograms](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53)

[3] [A good overview of Text To Speech](https://fritz.ai/speech-synthesis-with-deep-learning/)

[4] [Some papers on Speech Synthesis](https://paperswithcode.com/task/speech-synthesis)

**Pre-requisites:** Python Programming, Basic Understanding of Deep Learning and Neural Networks 

**Difficulty:** Hard

**Mentors:** Warren Jacinto, Veeransh Shah

**Domains:** Deep Learning, Natural Language Processing, Audio Processing

***

## GPT Reimagined: KANs vs MLPs
### Description
In this project, we aim to explore the effectiveness of Kolmogorov-Arnold Networks (KANs) as an alternative to traditional Multi Layer Perceptrons (MLPs) for implementing Generative Pretrained Transformers (GPTs). GPTs are a class of machine learning models known for their ability to generate natural language text and perform various natural language processing tasks. Traditionally, GPTs have been implemented using MLP architectures. However, KANs, a relatively new development, have shown promise in outperforming MLPs in certain tasks.

This project contributes to the ongoing research in machine learning architectures by providing empirical evidence on the efficacy of Kolmogorov-Arnold Networks as an alternative to traditional MLPs for implementing state-of-the-art language models like GPTs. The findings of this study can inform future developments in neural network architectures and guide the design of more efficient and effective models for natural language processing tasks.

Objectives:
- Implement GPT using the traditional MLP approach.
- Implement GPT using Kolmogorov-Arnold Networks (KANs).
- Compare the performance of GPT implemented with MLPs and KANs across various metrics, including but not limited to:
    - Language generation quality
    - Training speed
    - Model size
    - Resource utilization
- Provide a proof of principle for the performances of MLP-based GPTs versus KAN-based GPTs.

### References
[1] [MLP vs KAN](https://www.youtube.com/watch?v=-PFIkkwWdnM)

[2] [KANs explained](https://www.youtube.com/watch?v=7zpz_AlFW2w)

[3] [Original Research Paper proposing KANs](https://arxiv.org/pdf/2404.19756)

**Pre-requisites:** Python Programming, Basic Understanding of Machine Learning Algorithms and Neural Networks

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

**Pre-requisites:** Python Programming, Basic Understanding of Reinforcement Learning

**Difficulty:** Medium to Hard

**Mentors:** Vedant Mehra, Labhansh Naik

**Domains:** Game Development, Reinforcement Learning

***
## ChromaSight
### Description
ChromaSight is an innovative color correction tool designed to enhance visual experiences for individuals with color blindness. Our tool utilizes cutting-edge machine learning algorithms and natural language processing to provide personalized color correction solutions tailored to each user's specific color vision deficiencies.

The tool takes in images as input and processes them to enhance colors or apply color correction algorithms tailored for colorblindness. It analyzes the image and adjusts colors based on the type and severity of colorblindness. The tool provides an intuitive interface with options to upload images, select color correction modes, and view the corrected output.

Goals for this project are to:
- Incorporate accessibility features such as text-to-speech for navigation and colorblind-friendly UI design
- Utilize machine learning algorithms to automatically detect color blindness type and severity from user input or uploaded images.

The tool outputs corrected images that are optimized for colorblind individuals, ensuring better visibility and comprehension of visual content.

### References
[1] [A Color Guide for Color Blind People Using Image Processing and OpenCV](https://search.app/r2sUs5WKSrSa8m4i7)

[2] [Coloured Object Detection for Blind People Using CNN](https://search.app/tR6jCyaDcCpZeKpx7)

### Examples

![Sample Input](https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcR0aE17yppUAiK90Z6Gg47b0ik6Z2dLY4O5ndwKEkuiFhgSh-bk)

![Sample Output](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTf3Oj3cV4l6mKS5ZPYB8HHCgBxfLiEwKGbhrVraXDgsKDj49ui)

**Pre-requisites:** Python Programming, Basic Understanding of CNNs and Image Processing

**Difficulty:** Medium

**Mentors:** Aditi Dhumal, Anoushka Ruikar

**Domains:** Deep Learning, Computer Vision, Natural Language Processing

***

## Stock Transformers
### Description
In the modern capital market, the price of a stock is often considered to be highly volatile and unpredictable because of various social, financial, political and other dynamic factors that can bring catastrophic financial loss to the investors. This project aims to predict stock prices using transformer architecture by utilising the concept of time series forecasting.

The transformer model has been widely leveraged for natural language processing and computer vision tasks,but is less frequently used for tasks like stock prices prediction. The introduction of time2vec encoding to represent the time series features has made it possible to employ the transformer model for the stock price prediction. We aim to  leverage these two effective techniques to discover forecasting ability on volatile stock markets.

### References
[1] [Time series forecasting ](https://www.tableau.com/learn/articles/time-series-forecasting)

[2] [Transformer Architecture Article ](https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0)

[3] [Transformer Architecture 'Attention is all you need' paper](https://arxiv.org/pdf/1706.03762)

[4] [Time2Vec](https://youtu.be/1g9D9tW4AQk?si=3LOw6QlPJPoI-t_d) from 8:00 to 12:00

**Difficulty:** Medium

**Mentors:** Kindipsingh Malhi, Tvisha Vedant

**Domains:** Machine Learning, Deep Learning

***

## 🚗 2D Car Simulation with Genetic Algorithms 🧬
### Description

Welcome to the exciting world of 2D car simulations powered by physics engines and genetic algorithms! 🎉 In this project, we'll explore how these powerful tools can be combined to create a virtual playground where cars evolve and adapt to their environment. Our simulation will take place in a highly realistic 2D world governed by the laws of physics, featuring mind-blowing elements like collision detection, contact callbacks, convex polygons and circles, multiple shapes per body,  stable stacking, joints and constraints, and momentum and impulses, ensuring truly lifelike movements. Moreover, we'll implement our own physics engine, providing a deep understanding of the underlying mechanics and customization for our specific needs. Additionally, we'll harness the power of genetic algorithms (GAs) to create an intelligent and adaptive system that searches for the best car designs. GAs, inspired by natural selection and genetics, allow us to evolve solutions towards optimal performance. We'll start with a random initial population of car designs, evaluate their performance based on a fitness function (e.g., distance traveled, stability, etc.), and use genetic operations like crossover and mutation to create new generations. This iterative process continues until we find an optimal or satisfactory car design. By the end of this project, we will have a fully working simulation of 2D cars with all mentioned features and various parameters. 

![Box2D](https://lh6.googleusercontent.com/proxy/1zq8zn9ujbz9enGGvaGx9ILhQV2wHluTr-wT5IBh4VccGGouqsFJ-oc_qo4VJjY8SsUzwzqh_90)

# References

[1] [Basics Of Genetic Algorithms](https://www.youtube.com/watch?v=XP2sFzp2Rig)

[2] [Understanding Genetic Algorithms](https://www.geeksforgeeks.org/genetic-algorithms/)

[3] [Physics Enginge](https://box2d.org/)


**Pre-requisites**: Basic Understanding of Web Development, Mechanics, Linear Algebra  

**Difficulty**: Medium to Hard  

**Mentors**: Manas Bavaskar, Sharan Poojari  

**Domains**: Artificial Intelligence/Machine Learning, Computer Graphics and Simulation, Game Development  


***

## SmartMailGuard: AI-Powered Email Classification

![Alt Text](https://miro.medium.com/v2/resize:fit:1400/1*Fm58r_RQ53sEHfwFa28LpA.png)

### Description

In today's fast-paced digital world, our inboxes are bombarded with a relentless stream of emails, making it increasingly challenging to separate the essential from the irrelevant. Spam emails not only clutter our inboxes but also waste valuable time and pose security risks.

Our mission is to develop an AI-powered tool that can accurately distinguish between spam and legitimate emails, revolutionizing productivity and enhancing security. We'll embark on an exciting journey into probabilistic programming, starting with the Naive Bayes Algorithm to build the foundation from scratch. Moreover we'll delve into the world of advanced neural networks, utilizing Long Short-Term Memory (LSTM) networks for their ability to understand the context of emails over time. To push the boundaries even further, we'll harness the power of state-of-the-art architectures like transformers, which have set new benchmarks in natural language processing.

### References

[1] [Basics of Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

[2] [Introduction To Transformers](https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0)

[3] [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

[4] [Bayes Algorithm](https://petuum.medium.com/intro-to-modern-bayesian-learning-and-probabilistic-programming-c61830df5c50)

**Pre-requisites**: Python Programming, Basic Understanding of Neural Networks and Bayesian Algorithms

**Difficulty**: Medium to Hard

**Mentors**: Druhi Phutane, Raya Chakravarty

**Domains**: Deep Learning, Natural Language Processing, Probabilistic Programming


***
