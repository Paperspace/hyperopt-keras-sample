# Paperspace Hyperparameter Tuning for solving CIFAR-100 with a convolutional neural network (CNN) built with Keras and TensorFlow, GPU backend

[![Gradient](https://s3.amazonaws.com/ps.public.resources/img/gradient-badge-gradient-05-d-05-05.svg)](http://localhost:3002/console)



This project acts as both a tutorial and demo for using Paperspace Hyperparameter Tuning based on _Hyperopt_ with Keras, TensorFlow, and TensorBoard. Not only do we try to find the best hyperparameters for the given hyperspace, but we also represent the neural network architecture as hyperparameters that can be tuned. This serves to automate the process of searching for the best neural architecture configuration and hyperparameters.

Here, we are meta-optimizing a neural net and its architecture using the CIFAR-100 dataset, a computer vision task with 100 fine labeled items. This code could be easily transferred to another vision dataset or even to another machine learning task.

## How Hyperopt works

Hyperopt is a method for searching through a hyperparameter space. For example, it can use the Tree-structured Parzen Estimator (TPE) algorithm, which intelligently explores the search space while narrowing down to the best estimated parameters.

It is thus a good method for meta-optimizing a neural network. Whereas a neural network is an optimization problem that is tuned using gradient descent methods, hyperparameters cannot be tuned using gradient descent methods. That's where Hyperopt comes in and shines: it's useful not only for tuning hyperparameters like learning rate, but also for tuning more sophisticated parameters, and in a flexible way: it can change the number of layers of different types; the number of neurons in one layer or another; or even the type of layer to use at a certain place in the network given an array of choices, each themselves with nested, tunable hyperparameters.

This kind of Oriented Random Search is Hyperopt's strength, as opposed to a simpler Grid Search where hyperparameters are pre-established with fixed-step increases. Random Search for Hyperparameter Optimization has proven to be an effective search technique. The paper about this technique sits among the most cited deep learning papers. In summary, it is more efficient to randomly search through values and intelligently narrow the search space, rather than looping on fixed sets of hyperparameter values.

### How to define Hyperopt parameters

A parameter is defined with either a certain uniform range or a probability distribution, such as:

    hp.randint(label, upper)
    hp.uniform(label, low, high)
    hp.loguniform(label, low, high)
    hp.normal(label, mu, sigma)
    hp.lognormal(label, mu, sigma)

There are also a few quantized versions of those functions, which round the generated values at each step of "q":

    hp.quniform(label, low, high, q)
    hp.qloguniform(label, low, high, q)
    hp.qnormal(label, mu, sigma, q)
    hp.qlognormal(label, mu, sigma, q)

It is also possible to use a "choice" that can lead to hyperparameter nesting:

    hp.choice(label, ["list", "of", "potential", "choices"])
    hp.choice(label, [hp.uniform(sub_label_1, low, high), hp.normal(sub_label_2, mu, sigma), None, 0, 1, "anything"])

## Meta-optimize the neural network with Hyperopt

To run the hyperparameter search yourself, run: `python3 hyperopt_optimize.py`. You might want to look at `requirements.py` and install some of those dependencies manually in order to take advantage of GPU acceleration (such as by installing TensorFlow and Keras yourself, in particular).

Optimization results will continuously be saved in the `results/` folder (sort the files to get the best result as human-readable text).

Also, the results are pickled to `results.pkl` to be able to resume the TPE meta-optimization process later, which you can do simply by running the program again with `python3 hyperopt_optimize.py`.

If you want to learn more about Hyperopt, you'll probably want to [watch this video](https://www.youtube.com/watch?v=Mp1xnPfE4PY), made by the creator of Hyperopt. Also, if you want to run the model on a different dataset, such as CIFAR-10, you must edit the file `neural_net.py`.

It's possible that you may achieve better results than what's been achieved here in this repository. Pull requests / contributions are welcome! A couple of suggestions for finding interesting results: 1) try many different initializers for the layers; 2) add SELU activations. To restart the training with new or removed hyperparameters, it is recommended to delete existing results by running `./delete_results.sh`.

## The Deep Convolutional Neural Network Model

Below is a basic overview of the model. We implemented it in such a way that Hyperopt will try to change the shape of the layers and remove or replace some of them based on some pre-parameterized ideas that we're trying here. In this approach, Hyperopt changes a lot of parameters in addition to the learning rate.

```python

space = {
    # This loguniform scale will multiply the learning rate, so as to make
    # it vary exponentially, in a multiplicative fashion rather than in
    # a linear fashion, to handle his exponentialy varying nature:
    'lr_rate_mult': hp.loguniform('lr_rate_mult', -0.5, 0.5),
    # L2 weight decay:
    'l2_weight_reg_mult': hp.loguniform('l2_weight_reg_mult', -1.3, 1.3),
    # Batch size fed for each gradient update:
    'batch_size': hp.quniform('batch_size', 100, 700, 5),
    # Choice of optimizer:
    'optimizer': hp.choice('optimizer', ['Adam', 'Nadam', 'RMSprop']),
    # Coarse labels importance for weights updates:
    'coarse_labels_weight': hp.uniform('coarse_labels_weight', 0.1, 0.7),
    # Uniform distribution in finding appropriate dropout values, conv layers:
    'conv_dropout_drop_proba': hp.uniform('conv_dropout_proba', 0.0, 0.35),
    # Uniform distribution in finding appropriate dropout values, FC layers:
    'fc_dropout_drop_proba': hp.uniform('fc_dropout_proba', 0.0, 0.6),
    # Use batch normalization at more places?
    'use_BN': hp.choice('use_BN', [False, True]),

    # Use a first convolution that is special?
    'first_conv': hp.choice(
        'first_conv', [None, hp.choice('first_conv_size', [3, 4])]
    ),
    # Use residual connections? If so, how many more to stack?
    'residual': hp.choice(
        'residual', [None, hp.quniform(
            'residual_units', 1 - 0.499, 4 + 0.499, 1)]
    ),
    # Let's multiply the "default" number of hidden units:
    'conv_hiddn_units_mult': hp.loguniform('conv_hiddn_units_mult', -0.6, 0.6),
    # Number of conv+pool layers stacked:
    'nb_conv_pool_layers': hp.choice('nb_conv_pool_layers', [2, 3]),
    # Starting conv+pool layer for residual connections:
    'conv_pool_res_start_idx': hp.quniform('conv_pool_res_start_idx', 0, 2, 1),
    # The type of pooling used at each subsampling step:
    'pooling_type': hp.choice('pooling_type', [
        'max',  # Max pooling
        'avg',  # Average pooling
        'all_conv',  # All-convolutionnal: https://arxiv.org/pdf/1412.6806.pdf
        'inception'  # Inspired from: https://arxiv.org/pdf/1602.07261.pdf
    ]),
    # The kernel_size for convolutions:
    'conv_kernel_size': hp.quniform('conv_kernel_size', 2, 4, 1),
    # The kernel_size for residual convolutions:
    'res_conv_kernel_size': hp.quniform('res_conv_kernel_size', 2, 4, 1),

    # Amount of fully-connected units after convolution feature map:
    'fc_units_1_mult': hp.loguniform('fc_units_1_mult', -0.6, 0.6),
    # Use one more FC layer at output:
    'one_more_fc': hp.choice(
        'one_more_fc', [None, hp.loguniform('fc_units_2_mult', -0.6, 0.6)]
    ),
    # Activations that are used everywhere:
    'activation': hp.choice('activation', ['relu', 'elu'])
}

# Here is one possible outcome for this stochastic space; let's plot that:
space_base_demo_to_plot = {
    'lr_rate_mult': 1.0,
    'l2_weight_reg_mult': 1.0,
    'batch_size': 300,
    'optimizer': 'Nadam',
    'coarse_labels_weight': 0.2,
    'conv_dropout_drop_proba': 0.175,
    'fc_dropout_drop_proba': 0.3,
    'use_BN': True,

    'first_conv': 4,
    'residual': 4,
    'conv_hiddn_units_mult': 1.0,
    'nb_conv_pool_layers': 3,
    'conv_pool_res_start_idx': 0.0,
    'pooling_type': 'inception',
    'conv_kernel_size': 3.0,
    'res_conv_kernel_size': 3.0,

    'fc_units_1_mult': 1.0,
    'one_more_fc': 1.0,
    'activation': 'elu'
}

```

<p align="center">
  <img src="model_demo.png">
</p>

## Analysis of the hyperparameters

Here is an excerpt:

<p align="center">
  <img src="hyperparameters_scatter_matrix.png">
</p>

This could help to redefine the hyperparameters and to narrow them down successively, relaunching the meta-optimization on refined spaces.

## Best result

The final accuracy is 67.61% on average for the 100 fine labels, and is 77.31% on average for the 20 coarse labels.
These results are comparable to the ones in the middle of [this list](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d313030), under the CIFAR-100 section.
The only image preprocessing that we do is a random flip left-right.

### Best hyperspace found:

```python

space_best_model = {
    "coarse_best_accuracy": 0.7731000242233277,
    "coarse_best_loss": 0.8012041954994201,
    "coarse_end_accuracy": 0.7565,
    "coarse_end_loss": 0.9019438380718231,
    "fine_best_accuracy": 0.6761000108718872,
    "fine_best_loss": 1.3936876878738402,
    "fine_end_accuracy": 0.6549,
    "fine_end_loss": 1.539645684337616,
    "history": {...},
    "loss": -0.6761000108718872,
    "model_name": "model_0.676100010872_6066e",
    "real_loss": 3.018656848526001,
    "space": {
        "activation": "elu",
        "batch_size": 320.0,
        "coarse_labels_weight": 0.3067103474295116,
        "conv_dropout_drop_proba": 0.25923531175521264,
        "conv_hiddn_units_mult": 1.5958302613876916,
        "conv_kernel_size": 3.0,
        "conv_pool_res_start_idx": 0.0,
        "fc_dropout_drop_proba": 0.4322253354921089,
        "fc_units_1_mult": 1.3083964454436132,
        "first_conv": 3,
        "l2_weight_reg_mult": 0.41206755600055983,
        "lr_rate_mult": 0.6549347353077412,
        "nb_conv_pool_layers": 3,
        "one_more_fc": null,
        "optimizer": "Nadam",
        "pooling_type": "avg",
        "res_conv_kernel_size": 2.0,
        "residual": 3.0,
        "use_BN": true
    },
    "status": "ok"
}
```

### Plotting this best hyperspace's model:

<p align="center">
  <img src="model_best.png">
</p>

# To run it on a local environment

[Download](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) dataset for model training and put it to `ml_req` folder with name `cifar-100-python.tar.gz`.

Then from `local_infra` build and run with these commands:

```bash
docker-compose build
docker-compose up --scale hks_hyperopt_worker=2
```

Note: you should run hyperopt with at least 2 worker nodes for better reliability. (You might experience issues with unsaved evaluation results when running with a single worker node.)

**Important**
Remember to set available RAM for docker on mac to at least 4GB (1 GB per docker with assumption to run 2 hyperopt workers).

# To run it on Paperspace

## Using the Gradient CLI

You can run a hyperparameter tuning experiment on Paperspace using the [Gradient CLI](https://github.com/Paperspace/gradient-cli).

Assuming that you have configured an API Key for the Gradient CLI, enter:

```bash
gradient hyperparameters run \
  --name HyperoptKerasExperimentCLI1 \
  --projectId <your-project-id> \
  --tuningCommand 'make run_hyperopt' \
  --workerContainer tensorflow/tensorflow:1.13.1-gpu-py3 \
  --workerMachineType K80 \
  --workerCommand 'make run_hyperopt_worker' \
  --workerCount 2 \
  --workspaceUrl git+https://github.com/Paperspace/hyperopt-keras-sample
```

On successfully creating a hyperparameter experiment it should return something like this:

```
Hyperparameter created with ID: <experiment-id> and started
```

_Note: currently there not a CLI option to pass `experimentEnv` json to start hyperparameter with specific ENV values. However this functionality is available via the `/hyperopt/create_and_start` HTTP API call._

## Using the Paperspace Public API

Make a POST call to url: `https://services.paperspace.io/experiments/v1/hyperopt/create_and_start/`

With this call send headers:

- `x-api-key=<your-api-key>`
- `Content-Type=application/json`

Example json:

```json
{
  "name": "<your-experiment-name>",
  "tuningCommand": "make run_hyperopt",
  "workerCommand": "make run_hyperopt_worker",
  "workerContainer": "tensorflow/tensorflow:1.13.1-gpu-py3",
  "workerMachineType": "K80",
  "workerCount": 2,
  "projectId": "<your-project-id>",
  "workspaceUrl": "git+https://github.com/Paperspace/hyperopt-keras-sample",
  "experimentEnv": {
    "HKS_EPOCHS": 1,
    "HKS_MAX_EVALS": 2
  }
}
```
