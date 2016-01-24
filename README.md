# LSTM-Autoencoder

This project implements the Variational LSTM sequence to sequence architecture for a sentence auto-encoding task.
In general, I follow the paper "[Variational Recurrent Auto-encoders](http://arxiv.org/abs/1412.6581)" and "[Generating Sentences from a Continuous Space](http://arxiv.org/abs/1511.06349)". 
Most of the implementations about the variational layer are adapted from "[y0ast/VAE-torch](https://github.com/y0ast/VAE-Torch)".

## Descriptions
Following the above two papers, the variational layer is only added in between the last hidden state of the encoder and the first hidden state of the decoder, with the following steps:

1. Compute mean and variance of the posterior q from the last hidden state, with a 2-layer mlp encoder

2. Compute KLD loss between the estimated posterior q(z|x) and the enforced prior p(z) 

3. Collect a noise sample with reparameterization

4. Get the first hidden state of the decoder with a 2-layer mlp decoder

## Dependencies
This code requires [Torch7](http://torch.ch/) and [nngraph](http://github.com/torch/nngraph)

## Usage
On GPU: th LSTMAutoencoder.lua -gpuid 0


