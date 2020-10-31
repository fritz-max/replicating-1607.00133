# replicating-1607.00133
Differentially private classification of MNIST in PyTorch. This project replicates the results from [arXiv:1607.00133v2](https://arxiv.org/abs/1607.00133v2) [1].

## Project Description

The goal of the project is to train a simple classifier on the MNIST dataset, but in a differentially private way. A full explanation of the experiment and the used techniques can be found in [1]. As a quick overview, the experiment includes following steps: 

#### **1. Applying differentially private PCA to MNIST**

First, dimension reduction is performed using a differentially private version of PCA (Principal Components Analysis), adapted from [2]. This is mainly done to reduce training time, however it also increases model accuracy by around 2%. Furthermore, the accuracy is fairly stable accross different levels of noise applied to the PCA [1].      
This step is carried out in `src/ppca.py`. 

#### **2. Training the classifier**

The dimension-reduced MNIST dataset is used to train the model provided in `src/DPClassifier.py`. The training is carried out in `mnist_1607.00133.py`.  
The used differentially private optimizer (DPSGD) as well as the moments accountant are adopted from [3]. 

### Results
The results are included as plot in the `results/` folder and are shown here:
<img src="https://github.com/fritz-max/replicating-1607.00133/blob/main/results/accuracy_plot.png">

From Left to right the noise level increases, showing how this affects the accuracy of the model.

## Run the experiment yourself
You can replicate the results yourself by using the provided code. Clone the repo and install the requirements using

```sh
$ pip install -r requirements.txt
```
> Note: This command installs the library `pyvacy` [3] by cloning the repo and running its setup.py. 

After setup, run the experiment file
```sh
$ python mnist_1607.00133.py
``` 
**Disclaimer:** The combination of *differentially private SGD* and *moments accountant optimizer* used in this project require to compute the gradient for every individual sample. Therefore the training does not make use of GPU parallelization, making it **very slow** compared to todays standards of training neural networks.  

## References

[1] Martín Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, Li Zhang. *Deep Learning with Differential Privacy*. [arXiv:1607.00133v2](https://arxiv.org/abs/1607.00133v2)

[2] IBM. [*IBM Differential Privacy Library*](https://github.com/zhehedream/COEN281).

[3] Chris Waites. [*PyVacy: Privacy Algorithms for PyTorch*](https://github.com/ChrisWaites/pyvacy).

## Licenses 
?
