<div align="center">
  <h1><b>Genetic Algorithm for Neural Network Weight Optimisation</b></h1>
  <img src="https://images.weserv.nl/?url=avatars.githubusercontent.com/u/61354833?v=4&h=100&w=100&fit=cover&mask=circle&maxage=7d">
  <p><b>Code by <a href="https://github.com/TomMakesThings">TomMakesThings</a></b></p>
  <p><b><sub>November 2021</sub></b></p>
</div>

---

# **About**

### **Problem Definition**
The aim of this project is to train a feed-forward multi-layer perceptron network to approximate the following function:

$y = sin(2x_{1} + 2.0) \text{+} cos(0.5x_{2}) + 0.5$

$x_{1}, x_{2} \in [0, 2\pi]$

The neural network is fully connected and has two inputs, two bias neurons, six hidden neurons and one output. Sigmoid activation functions are used in the hidden neurons, and a linear activation function in the output neuron.

<img src="https://github.com/TomMakesThings/Computational-Intelligence-Neural-Network/blob/main/Images/Neural-Network.png">

### **Implementation**

**Training and Testing Data**

21 random values between $[0, 2\pi]$ are generated for $x_{1}$ and $x_{2}$ and added to lists retrospectively. These values are used to calculate a list of corresponding $y$ values. 11 sets of $x_{1}$, $x_{2}$, $y$ values are written to file `train.dat`, while the remaining values are written to file `test.dat`. 

<br>

**Encoding**

A binary coded genetic algorithm is used for optimising the weights of the neural network. Every individual in the population is initialised with 25 random 15-bit decision variables with each representing a network weight. This includes the weights from the bias nodes. The number of decision variables was calculated using the following formula:

$(\text{INPUTS} \text{+} 1) \times HIDDEN \text{+} (HIDDEN + 1) \times OUTPUTS$

Therefore, the number of weights in the network is:

$(2 + 1) \times 6 + (6 + 1) = 25$

Consequently, the total number of bits to encode the weights per individual is:

$15 \times 25 = 375$

<br>

**Population Initialisation and Evalution**

100 individuals are generated in each population. For each, their decision variables are converted from binary to their real value between the range $[-10, 10]$. The fitness function sets individual’s real-valued weights as the weights of the network and calculates the mean squared error. The loss value is then set as the individual’s fitness.

<br>

**Mate Selection and Crossover**

After evaluating fitness, individuals are randomly selected as parent pairs. 200 offspring are produced by applying uniform crossover with cross probability 0.95. This high cross probability creates variety within the weights. Similarly, with uniform crossover, if the two parents are different, the offspring are unlikely to be the same as either parent. Both factors ensure that the population maintains high genetic diversity preventing the population getting stuck at a local minimum.

<br>

**Mutation**

Point mutation is applied to each newly generated offspring with probability:

$p = \frac{1}{\text{n bits} \times \text{n decision variables}}$

This means that on average, there will be one mutation per chromosome. Keeping this value low prevents the population from changing too rapidly to allow convergence to an optimum weight set. However, a small amount of mutation helps introduce new variation that may not be possible from parent crossover alone.

<br>

**Selection**

The offspring's fitness is evaluated and k-elitism is used to select the top 100 offspring pass on to the next generation. Selecting the best individuals speeds up convergence over other methods such as rank proportionate.

<br>

**Running the Neural Network**

The genetic algorithm is set to run 40 times to optimise the weights of the network. In each generation, if no lifetime learning method is set then an individuals’ fitness is the MSE on training data. After training within each generation, the individual with the highest MSE is recorded and this individual is then evaluated against testing data. The MSE of these best individuals are then plotted.

The maximum number of generations has been set as 40 as the MSE generally converges after this number (left graph). If further generations are run, this will lead to overfitting of training data (right graph).

<img width=600 src="https://github.com/TomMakesThings/Computational-Intelligence-Neural-Network/blob/main/Images/MSE-No-Learning.png">

<br>

**Lamarckian Learning Approach**

Genetic algorithms are great at exploring a search space, but not good at converging. Therefore a memetic algorithm can be introduced to extend the genetic algorithm to use a local search.

Instead of evaluating the fitness of an individual’s decision variables at initialisation, fitness can be assigned after lifetime learning through local search. During local search, the individual’s decision variables are set as the weights for the network. For 30 iterations, the network is trained using Rprop, a method that uses the gradient to update the network weights. After the local search is complete, the updated weights and MSE are returned.

The new weights are converted from real valued to binary by reversing the formulae for conversion from binary to real valued. The individual’s corresponding decision variables are updated with these new weights and the new fitness assigned. As the individual's decision variables are updated and passed to offspring, this is Lamarckian learning.

<img width=600 src="https://github.com/TomMakesThings/Computational-Intelligence-Neural-Network/blob/main/Images/Lamarckian.png">

<br>

**Baldwinian Learning Approach**

With the Baldwinian learning approach, Rprop learning is used to improve fitness, as implemented with Lamarckian learning. However, an individual’s decision variables are not changed after learning and therefore are not passed onto offspring. By not inheriting the improved weights, it takes longer for individuals in the Baldwinian population to improve and converge to an optimum.

<img width=600 src="https://github.com/TomMakesThings/Computational-Intelligence-Neural-Network/blob/main/Images/Baldwinian.png">

# Instructions to Run the Code
1. [Click here](https://github.com/TomMakesThings/Computational-Intelligence-Neural-Network/archive/refs/heads/main.zip) to download the repository contents
2. Extract the [Jupyter notebook](https://github.com/TomMakesThings/Computational-Intelligence-Genetic-Algorithm/blob/main/Genetic_Algorithm.ipynb) from the ZIP
3. Import the notebook into [Colab](https://colab.research.google.com/)
