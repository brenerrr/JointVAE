# Joint VAE with Keras

This repository implements the algorithm from the paper [Learning Disentangled Joint Continuous and Discrete Representations](https://arxiv.org/abs/1804.00104). As stated in the abstract, the main objective is to "learn a disentangled and interpretable jointly continuous and discrete representations in an unsupervised manner".

Putting in simpler words, we want to force each dimension of the latent space of a VAE to change something different from the other ones. If we consider images of digits, then hopefully changing one dimension would alter the thickness of the strokes, while another would change the angle of the digit.

The way the author achieved this was by initially forcing the latent space to look like the normal and gumbell distributions for continuous and discrete variables, respectively. Then as training goes on, the optimization tries to make the latent space be some kind of distribution within a certain KL divergence from the initial ones. If you would like to know more on why this procedure achieves disentanglement, I suggest reading the paper [Understanding disentangling in β-VAE](https://arxiv.org/abs/1804.03599).

# Setup

All necessary packages can be installed with

```
pip install -r requirements.txt
```

# Training

A VAE can be trained with hyperamaraters stored in `hyperparameters.json` with the command

```
python train_model.py
```

All relevant statistics are automatically logged and can be viewed with

```
tensorboard --logdir logs
```


# Visualization

An interactive app made with `Dash` can be launch with

```
python app.py -model <MODEL NAME> -i <ORDER OF CONTINUOUS LATENT>
```

In order to know which continuous latent dimensions actually matter, check the tensorboard logs and see which ones have the highest KL divergence.

# Trained Model

You can find the weights of a fully trained model named `trained_model` in this repository. You can play with it with

```
python app.py -model trained_model -i 3 8 6 1 0
```