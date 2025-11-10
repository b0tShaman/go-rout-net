package main

import (
	"math"
	"math/rand"
)

const (
	leak            = 0.01
	channelCapacity = 1
)

var activationMap = map[string]struct {
	fn ActivationFunc
	df ActivationFunc
}{
	"linear":    {linear, dfLinear},
	"sigmoid":   {sigmoid, dfSigmoid},
	"relu":      {relu, dfRelu},
	"leakyrelu": {leakyRelu, dfLeakyRelu},
	"tanh":      {tanh, dfTanh},
	"gelu":      {gelu, dfGelu},
}

type ActivationFunc func(float64) float64
type InitializerFunc func([]float64, int, int)

type Neuron struct {
	Weights []float64
	Bias    float64
	LR      float64

	ErrsToPrev   []chan float64 // send error backward to all neurons of previous layer connected to this neuron : write
	ErrsFromNext []chan float64 // receive error from layer in front, from each neuron to which this neuron is connected to : read

	OutsFromPrev []chan float64 // activated outputs from prev layer, outputs from all neurons of previos layer connected to this neuron  : read
	InsToNext    []chan float64 // activated outputs to next layer, send output to all the neurons in the next layer to which this neuron is connected to : write

	BatchSize   int         // Stores the batch size
	BatchGradsW [][]float64 // To accumulate dL/dW for each sample in the batch
	BatchGradB  []float64   // To accumulate dL/dB for each sample in the batch
	BatchCount  int         // To track samples processed in current batch

	// A channel to safely signal config changes to the backward worker
	ConfigUpdate chan NeuronCfg // Channel to send new BatchSize
}

type NeuronCfg struct {
	LR        float64
	BatchSize int
	Ack       chan struct{} // signal completion
}

// Information stored between forward and backward
type Knowledge struct {
	Input []float64
	T     float64
}

func NewNeuron(errsToPrev, outsFromPrev, errsFromNext, insToNext []chan float64, f, df ActivationFunc, init InitializerFunc) *Neuron {
	n := &Neuron{
		Weights:      make([]float64, len(outsFromPrev)),
		Bias:         0.0,
		ErrsToPrev:   errsToPrev,
		ErrsFromNext: errsFromNext,
		OutsFromPrev: outsFromPrev,
		InsToNext:    insToNext,
		ConfigUpdate: make(chan NeuronCfg), // Buffered channel for new batchSize
	}

	// To calculate gradient averages for weight updates
	avgBiasGrad := 0.0
	avgWeightsGrads := make([]float64, len(n.Weights))

	// Init weights
	init(n.Weights, len(n.OutsFromPrev), len(n.InsToNext))

	// Channel for internal forward/backward knowledge
	ch := make(chan Knowledge, channelCapacity)

	// Forward activation closure
	Activate := func(input []float64) float64 {
		t := n.Bias
		for i, val := range input {
			t += val * n.Weights[i]
		}

		// Store forward knowledge for backward use
		select {
		case ch <- Knowledge{Input: input, T: t}:
		default:
			// log.Println("Failed to store knowledge")
		}
		return f(t)
	}

	// Launch forward worker goroutine
	go func() {
		for {
			in := make([]float64, len(n.OutsFromPrev))
			for i := range n.OutsFromPrev {
				in[i] = <-n.OutsFromPrev[i]
			}

			out := Activate(in)
			for i := range len(n.InsToNext) {
				n.InsToNext[i] <- out
			}
		}
	}()

	// Launch backward worker goroutine
	go func() {
		for {
			select {
			case cfg := <-n.ConfigUpdate:
				n.LR = cfg.LR
				n.BatchSize = cfg.BatchSize
				n.BatchCount = 0 // Reset counter
				n.BatchGradsW = make([][]float64, cfg.BatchSize)
				n.BatchGradB = make([]float64, cfg.BatchSize)
				for i := range cfg.BatchSize {
					n.BatchGradsW[i] = make([]float64, len(n.Weights))
				}
				close(cfg.Ack)

			// wait for forward pass to store state
			case kw := <-ch:
				// compute local gradient
				grad := df(kw.T)

				// wait for all incoming errors from front layer
				errFront := 0.0
				for _, outCh := range n.ErrsFromNext {
					errFront += <-outCh
				}

				// send error backward = w_current * grad(t) * errFront
				for i := range n.ErrsToPrev {
					n.ErrsToPrev[i] <- grad * n.Weights[i] * errFront
				}

				if n.BatchSize == 1 {
					for i := range n.Weights {
						n.Weights[i] -= n.LR * grad * kw.Input[i] * errFront
					}
					n.Bias -= n.LR * grad * errFront
					continue
				}

				// --- ACCUMULATION STEP (per sample) ---
				sampleIndex := n.BatchCount

				n.BatchGradB[sampleIndex] = grad * errFront
				for i := range n.Weights {
					n.BatchGradsW[sampleIndex][i] = grad * kw.Input[i] * errFront
				}

				n.BatchCount++
				// --- END ACCUMULATION STEP ---

				// --- BATCH UPDATE STEP ---
				if n.BatchCount == n.BatchSize {
					// 1. Calculate averages
					for i := range n.BatchSize {
						avgBiasGrad += n.BatchGradB[i]
						for j := range n.Weights {
							avgWeightsGrads[j] += n.BatchGradsW[i][j]
						}
					}
					avgBiasGrad /= float64(n.BatchSize)
					for j := range n.Weights {
						avgWeightsGrads[j] /= float64(n.BatchSize)
					}

					// 2. Update weights and bias (using the average gradients)
					for i := range n.Weights {
						n.Weights[i] -= n.LR * avgWeightsGrads[i]
					}
					n.Bias -= n.LR * avgBiasGrad

					// 3. Reset batch counter
					n.BatchCount = 0
					avgBiasGrad = 0.0
					for j := range n.Weights {
						avgWeightsGrads[j] = 0.0
					}
				}
				// --- END BATCH UPDATE STEP ---
			}
		}
	}()

	return n
}

// Activations and Derivatives
func linear(x float64) float64 {
	return x
}

func dfLinear(x float64) float64 {
	return 1
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func dfSigmoid(x float64) float64 {
	s := sigmoid(x)
	return s * (1 - s)
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func dfRelu(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

func leakyRelu(x float64) float64 {
	if x > 0 {
		return x
	}
	return leak * x
}

func dfLeakyRelu(x float64) float64 {
	if x > 0 {
		return 1
	}
	return leak
}

func tanh(x float64) float64 {
	return math.Tanh(x)
}

func dfTanh(x float64) float64 {
	t := math.Tanh(x)
	return 1 - t*t
}

func gelu(x float64) float64 {
	return 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*math.Pow(x, 3))))
}

func dfGelu(x float64) float64 {
	// Approx derivative of GELU
	const c = 0.0356774
	const b = 0.797885
	t := math.Tanh(b * (x + c*math.Pow(x, 3)))
	return 0.5 * (1 + t + x*(1-t*t)*(b+3*b*c*x*x))
}

// Weight Initializers
func XavierInit(weights []float64, fanIn, fanOut int) {
	limit := math.Sqrt(6.0 / float64(fanIn+fanOut))
	for i := range weights {
		weights[i] = rand.Float64()*(2*limit) - limit
	}
}

func XavierNormal(weights []float64, fanIn, fanOut int) {
	std := math.Sqrt(2.0 / float64(fanIn+fanOut))
	for i := range weights {
		weights[i] = rand.NormFloat64() * std
	}
}

func HeNormal(weights []float64, fanIn, fanOut int) {
	std := math.Sqrt(2.0 / float64(fanIn))
	for i := range weights {
		weights[i] = rand.NormFloat64() * std
	}
}

func HeUniform(weights []float64, fanIn, fanOut int) {
	limit := math.Sqrt(6.0 / float64(fanIn))
	for i := range weights {
		weights[i] = rand.Float64()*(2*limit) - limit
	}
}

func Random(weights []float64, fanIn, fanOut int) {
	for i := range weights {
		weights[i] = rand.Float64()*0.2 - 0.1
	}
}
