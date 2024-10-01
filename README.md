# Discrete-VAE_Perturbation-Models

This repository shows an experiment from the paper "On The Representation Properties Of The Perturb-Softmax And The Perturb-Argmax Probability Distribution" https://arxiv.org/abs/2406.02180

It demonstrates the advantage of the Gaussian-Softmax over the commonly used Gumbel-Softmax. It exhibits that, compared to the Gumbel-Softmax, the Gaussian-Softmax enjoys a faster convergence rate and better approximate discrete distributions. Please refer to the paper for a theoretical analysis of the properties of softmax (and argmax) perturbation models

## Experiment description
We compared the training ELBO-based loss of categorical Variational-Autoencoders for N=10 variables, each is a K-dimensional categorical variable, K =[10,30,50] on the binarized MNIST, the Fashion-MNIST, and the Omniglot datasets for different smooth perturbation distributions.  
The architecture consists of an encoder of X -> FC(300) -> ReLU -> N*K, and a matching decoder N*K -> FC(300) -> ReLU ->X. The loss is the traditional composition of the reconstruction error and the KL divergence. 

A fair comparison between Perturb-Softmax models with different perturbation distributions requires temperature selection for
each model. The temperature is a hyperparameter that affects the models' performance and, thus should be chosen with cross-validation. We compare the loss obtained by these Pertub-Softmax with temperature model  models for a range of temperatures = {0.01, 0.03, 0.07, 0.1, 0.25, 0.4, 0.5, 0.67, 0.85, 1.0}. The test set loss is calculated for each temperature with the model achieving the lowest loss on the validation set. 

## Results
Results show that by comparing the best-performing temperature-based models, the Normal-Softmax model consistently achieves the lowest test set loss for all datasets. 
Next, we analyze the training convergence when propagating gradients with the Normal-Softmax or the Gumbel-Softmax of these models for temperature equals $1$.
The results below on the Omniglot dataset K = 10, 30,50 show that the former achieves better and faster learning convergence in all experiments.


<img src="https://github.com/user-attachments/assets/905b256d-fdd7-41ea-9c21-ddb3c7d0a03d" width="200">

<img src="https://github.com/user-attachments/assets/d4d8eb9c-3f28-400a-82d3-15344f566e7d" width="200">

<img src="https://github.com/user-attachments/assets/ccb4b361-ef9f-4e6b-858f-bec7119a4a3b" width="200">

### To run the code
Run Perturb_sm.py with the relevant arguments. For example, 
python Perturb_sm.py --perturb Normal --K 10 --ds mnist

#### Acknowledgment
This code is based on <a href="https://www.w3schools.com](https://github.com/GuyLor/Direct-VAE)">Direct-VAE</a>



