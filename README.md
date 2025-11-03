# go-concurrent-nn
A from-scratch, fully concurrent neural network simulator in Go — each neuron runs two goroutines (one for feedforward, one for backprop) and communicates via channels, mimicking asynchronous biological signalling.
