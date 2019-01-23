package nn

import (
	"fmt"
	"math"
	"math/rand"
)

var eta = 0.5
var alpha = 0.05

type Connection struct {
	Weight      float64
	DeltaWeight float64
}

type Neuron struct {
	Index    int
	Value    float64
	Gradient float64
	Weights  []Connection
}

func newNeuron(index int, nextLayerSize int) *Neuron {
	weights := make([]Connection, nextLayerSize)
	for i := range weights {
		weights[i].DeltaWeight = rand.Float64()
		weights[i].Weight = rand.Float64()
	}
	n := &Neuron{
		Index:    index,
		Value:    0,
		Gradient: 0,
		Weights:  weights}
	return n
}

func (n *Neuron) FeedForward(prevLayer Layer) {
	var sum float64
	for _, nn := range prevLayer {
		sum += nn.Value * nn.Weights[n.Index].Weight
	}
	n.Value = math.Tanh(sum)
}

func (n *Neuron) CalcOutputGradients(target float64) {
	delta := target - n.Value
	n.Gradient = delta * transferDerivative(n.Value)
}

func transferDerivative(value float64) float64 {
	return 1.0 - value*value
}

func (n *Neuron) CalcHiddenGradients(nextLayer Layer) {
	var sum float64
	for i, nn := range nextLayer {
		sum += n.Weights[i].Weight * nn.Gradient
	}
	n.Gradient = sum * transferDerivative(n.Value)
}

func (n *Neuron) UpdateInputWeights(prevLayer Layer) {
	for _, nn := range prevLayer {
		oldDeltaWeight := nn.Weights[n.Index].DeltaWeight
		newDeltaWeight := eta*nn.Value*n.Gradient + alpha*oldDeltaWeight

		nn.Weights[n.Index].DeltaWeight = newDeltaWeight
		nn.Weights[n.Index].Weight += newDeltaWeight
	}
}

func (n *Neuron) String() string {
	return fmt.Sprintf("%v", n.Value)
}
