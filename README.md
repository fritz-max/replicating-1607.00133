# replicating-1607.00133
Differentially private classification of the MNIST dataset from [arXiv:1607.00133v2](https://arxiv.org/abs/1607.00133v2) [1] in PyTorch.

# Project Description

This project aims to replicate the results of the mentioned paper on the MNIST dataset as close as possible, primarily to experiment with differential privacy in PyTorch. 
This includes following steps

**Applying differentially private PCA to MNIST** 

The functions in `src/pca.py` file download the MNIST dataset from torchvision and apply differentially private Principal Components Analysis [2] to reduce input dimensionality from 784 to 60.  
The processed Training and Testing Dataset are also provided in the XY folder (Really?) 

**Training a simple Neural Network on the data**

The data is used to train the model provided in `src/DPClassifier.py`, which is done in `mnist_1607.00133.py`. The differentially private optimizer and moments accountant in use are adopted from [3]. 


# Results
Can be seen in results folder.

  - [ ] TODO -> include plots

include plots here -> show them and discuss a little 

# Run the experiment yourself
You can replicate the results yourself by using the provided code. 

## Prerequisites
Clone the repo and install the requirements using

```sh
$ pip install -r requirements.txt
```
Refer to the `requirements.txt` file for the list of dependencies. 

- mention that pyvacy is being installed by cloning the repo (link) and running setup.py

## Execution
Once set up, the experiment can be run using 
```sh
$ python mnist_1607.00133.py
``` 
**Disclaimer:** The combination of *differentially private SGD* and *moments accountant optimizer* used in this project require to compute the gradient for every individual sample. Therefore the training does not make use of GPU parallelization, making it **very slow** compared to todays standards of training neural networks.  


# References

[1] Martín Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, Li Zhang. *Deep Learning with Differential Privacy*. [arXiv:1607.00133v2](https://arxiv.org/abs/1607.00133v2)

[2] IBM. [*IBM Differential Privacy Library*](https://github.com/zhehedream/COEN281).

[3] Chris Waites. [*PyVacy: Privacy Algorithms for PyTorch*](https://github.com/ChrisWaites/pyvacy).



## Licenses 
?


#
## Todolist wednesday
- [X] Which plots?
  - 1.1 - 1.3 -> accuracy for different levels of noise (fig. 3)
  - 2 -> accuracy for epsilon, deltas (fig. 4)
  - 3 -> influence of epsilon on pca 
- [ ] Fix up github repo
  - [ ] readme
    - [ ] references, used work
    - [ ] dependencies, setup and how to run
  - [ ] code comments 
  - [ ] requirements.txt
  - [ ] code 
    - [x] private pca code
    - [ ] plots
    - [x] baseline model
  - [ ] data folder
    - [ ] results in a file 
    - [ ] pca mnist (datasets) in a file
    - [ ] plots
