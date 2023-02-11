# Gaussian-Processes
In this project I aim to tune the hyper-parameters for global and local kernel for a Gaussian process (GP) to predict the motion of a sensor. 
I have used GP to fit target data as it can give a non parametric representation of data processes if we assume thatdata is sampled from a multivariate GP. Moreover only the first and
Using different hyper-parameters we can define kernel functions and we can optimise these parameters to fit the data. The kernel function defines the 
co-variance and how it acts prior to the GP.
The data given to us comprises 50 sensors that were used to monitor 3D human movements. 
I chose the AG data set and the sensor 15 due to computational constraints. 
The data set contains the motion data for a certain part of the body for a given space time.
The goal is to represent a subjects motion as a GP. 
I have chosen the 15th marker placed on the right hand due to computational constraints.
