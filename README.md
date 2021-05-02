# Advanced Topics in Deep Learning course project

This project is intended for the *Advanced Topics in Deep Learning* course, as a part of the M2 Masters curriculum **Data Sciences** of *Institut Polytechnique de Paris*.

The authors of this project are:
 * Stéphane Bereux
 * Lotfi Kobrosly

## Structure/ How tu use
The code available under the directory `./code` is run by simply running `python main.py` in the command line. Then the user is asked to choose between which dataset to inspect (MNIST or CIFAR10) and the model to use (LeNet or Convolutional). The code is set to run on GPUs if available.
The model trains until early stopping occurs, then its weights are pruned by the lowest magnitude elements, to have the kept ones restored to their initial value and then have another training session, while saving accuracies and early-stopping iterations.

The files in `./code` are as follows:
* **main.py** contains the code to launch the desired tests
* **nets.py** contains the classes used to define the models that are used
* **utils.py** contains the training monitor provided by *pytorch.ignite* to apply early-stopping

In `./networks` we have:
* **networks.json** which contains the structures of the Lenet and Convolutional networks
* **optimal_params.json** contains optimal parameters obtained from the article related to this project

`./data` contains the MNIST and CIFAR10 datasets loaded through `pytorch` (not available on the repository).

## Important Note:
While the code gives the option of using *Conv4* and *Conv6* networks on the MNIST and CIFAR10 datsets, it is not possible to run them, as the convolutional layers present in these two networks provide an image of size 1 x 1 to the fully-connected layers that come after them, which implies a significant loss of information. However, we may add other datasets in the future.
