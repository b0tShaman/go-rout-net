package main

import (
	"log"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	X, Y, err := LoadCSV("mnist_train.csv")
	if err != nil {
		log.Println("Error:", err)
		return
	}
	MinMaxNormalize(X)

	inputDim := len(X[0])
	numSamples := len(X)

	log.Printf("Loaded dataset: %d samples, %d input features\n", numSamples, inputDim)

	nw := NewNetwork(
		Dense(64, InputDim(inputDim), Activation("relu"), Initializer("he")),
		Dense(32, Activation("relu")),
		Dense(16, Activation("relu")),
		Dense(10),
	)

	// Try to load previous weights if file exists
	if err := nw.LoadWeights("weights.json"); err != nil {
		log.Println("Error loading weights:", err)
	}

	// Setup Ctrl+C handler
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-c
		log.Println("\nSaving weights before exit...")
		if err := nw.SaveWeights("weights.json"); err != nil {
			log.Println("Error saving weights:", err)
		} else {
			log.Println("Weights saved successfully.")
		}
		os.Exit(0)
	}()

	cfg := TrainingConfig{
		Epochs:       10,
		BatchSize:    1,
		LearningRate: 0.01,
		LossFunction: CATEGORICAL_CROSS_ENTROPY,
		KClasses:     10, // For CATEGORICAL_CROSS_ENTROPY (Softmax Output).
		VerboseEvery: 1,
	}

	// Train
	nw.Train(X, Y, cfg)

	// Evaluate
	X, Y, err = LoadCSV("mnist_test.csv")
	if err != nil {
		log.Println("Error:", err)
		return
	}
	MinMaxNormalize(X)
	loss, acc := nw.Evaluate(X, Y, cfg)
	log.Printf("Final Evaluation: Loss=%.6f, Accuracy=%.2f%%", loss, acc)

	// Predict
	test, err := convertJpg1D("predict.jpg")
	if err != nil {
		panic("convertJpg1D failed")
	}
	NormalizeSampleMinMax(test)
	pred := nw.Predict(test, cfg)
	log.Printf("Predicted Output: %.4f | Actual Output: %.4f\n", pred, 5.0)

	// Save weights
	if err := nw.SaveWeights("weights.json"); err != nil {
		log.Println("Error saving weights:", err)
	} else {
		log.Println("Weights saved successfully.")
	}
}
