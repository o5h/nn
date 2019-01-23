package nn

import (
	"fmt"
	"math"
)

type Layer []*Neuron

func newLayer(size int, nextLayerSize int) Layer {
	layer := make([]*Neuron, size)
	for i := range layer {
		layer[i] = newNeuron(i, nextLayerSize)
	}
	return layer
}

type Network struct {
	Layers []Layer
	Error  float64
}

func New(topology ...int) *Network {
	n := new(Network)
	layersNumber := len(topology)
	n.Layers = make([]Layer, layersNumber)
	for i, layerSize := range topology {
		nextLayerSize := 0
		if i+1 < layersNumber {
			nextLayerSize = topology[i+1]
		}
		layer := newLayer(layerSize, nextLayerSize)
		n.Layers[i] = layer
	}
	n.Error = 0.5
	return n
}

func (n *Network) GetResults() []float64 {
	lastLayer := n.Layers[len(n.Layers)-1]
	result := make([]float64, len(lastLayer))
	for i, n := range lastLayer {
		result[i] = n.Value
	}
	return result
}

func (n *Network) FeedForward(input []float64) {
	firstLayer := n.Layers[0]
	for i := range input {
		firstLayer[i].Value = input[i]
	}
	for layerIndex := 1; layerIndex < len(n.Layers); layerIndex++ {
		for _, neuron := range n.Layers[layerIndex] {
			neuron.FeedForward(n.Layers[layerIndex-1])
		}
	}
}

func (n *Network) BackPropagation(target []float64) {
	outputLayer := n.Layers[len(n.Layers)-1]
	n.Error = 0
	for i, nn := range outputLayer {
		delta := target[i] - nn.Value
		n.Error += delta * delta
	}
	n.Error = n.Error / float64(len(outputLayer))
	n.Error = math.Sqrt(n.Error)

	for i, nn := range outputLayer {
		nn.CalcOutputGradients(target[i])
	}

	for layerNumber := len(n.Layers) - 2; layerNumber > 0; layerNumber-- {
		hiddenLayer := n.Layers[layerNumber]
		nextLayer := n.Layers[layerNumber+1]
		for _, nn := range hiddenLayer {
			nn.CalcHiddenGradients(nextLayer)
		}
	}

	for layerNumber := len(n.Layers) - 1; layerNumber > 0; layerNumber-- {
		layer := n.Layers[layerNumber]
		prevLayer := n.Layers[layerNumber-1]
		for _, nn := range layer {
			nn.UpdateInputWeights(prevLayer)
		}
	}
}

func (n *Network) String() string {
	return fmt.Sprintf("%v\n", n.Layers)
}
