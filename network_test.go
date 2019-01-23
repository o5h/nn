package nn_test

import (
	"math"
	"testing"

	"github.com/o5h/nn"
)

type TestData struct {
	Input  []float64
	Output []float64
}

// var allTestData = []TestData{
// 	{[]float64{0, 0, 0, 0}, []float64{0.0}},
// 	{[]float64{0, 0, 0, 1}, []float64{0.1}},
// 	{[]float64{0, 0, 1, 0}, []float64{0.2}},
// 	{[]float64{0, 0, 1, 1}, []float64{0.3}},
// 	{[]float64{0, 1, 0, 0}, []float64{0.4}},
// 	{[]float64{0, 1, 0, 1}, []float64{0.5}},
// 	{[]float64{0, 1, 1, 0}, []float64{0.6}},
// 	{[]float64{0, 1, 1, 1}, []float64{0.7}},
// 	{[]float64{1, 0, 0, 0}, []float64{0.8}}}

// var allTestData = []TestData{
// 	{[]float64{0, 0, 0, 0}, []float64{0.0}},
// 	{[]float64{0, 0, 0, 1}, []float64{0.1}},
// 	{[]float64{0, 0, 1, 0}, []float64{0.1}},
// 	{[]float64{0, 0, 1, 1}, []float64{0.3}},
// 	{[]float64{0, 1, 0, 0}, []float64{0.1}},
// 	{[]float64{0, 1, 0, 1}, []float64{0.2}},
// 	{[]float64{0, 1, 1, 0}, []float64{0.2}},
// 	{[]float64{0, 1, 1, 1}, []float64{0.3}},
// 	{[]float64{1, 0, 0, 0}, []float64{0.1}}}

var allTestData = []TestData{

	{[]float64{0, 0, 0, 0}, []float64{0.0}},
	{[]float64{0, 0, 0, 1}, []float64{0.1}},
	{[]float64{0, 0, 1, 0}, []float64{0.2}},
	{[]float64{0, 0, 1, 1}, []float64{0.3}},

	{[]float64{0, 1, 0, 0}, []float64{0.1}},
	{[]float64{0, 1, 0, 1}, []float64{0.2}},
	{[]float64{0, 1, 1, 0}, []float64{0.3}},
	{[]float64{0, 1, 1, 1}, []float64{0.4}},

	{[]float64{1, 0, 0, 0}, []float64{0.2}},
	{[]float64{1, 0, 0, 1}, []float64{0.3}},
	{[]float64{1, 0, 1, 0}, []float64{0.4}},
	{[]float64{1, 0, 1, 1}, []float64{0.5}},

	{[]float64{1, 1, 0, 0}, []float64{0.3}},
	{[]float64{1, 1, 0, 1}, []float64{0.4}},
	{[]float64{1, 1, 1, 0}, []float64{0.5}},
	{[]float64{1, 1, 1, 1}, []float64{0.6}},
}

func TestNetwork(t *testing.T) {
	network := nn.New(4, 8, 1)
	generation := 0
	for network.Error > 0.001 {
		for _, testData := range allTestData {
			network.FeedForward(testData.Input)
			network.BackPropagation(testData.Output)
		}
		generation++
		t.Log("Error:", network.Error)
	}

	t.Log("Generation", generation)
	for _, testData := range allTestData {
		network.FeedForward(testData.Input)
		result := network.GetResults()

		actual := result[0]
		expected := testData.Output[0]
		delta := math.Abs(actual - expected)
		t.Log("result", actual, expected, delta)
		if delta > 0.001 {
			t.Fail()
		}
	}
}

func Benchmark481(b *testing.B) {
	network := nn.New(4, 8, 1)
	for n := 0; n < b.N; n++ {
		for _, testData := range allTestData {
			network.FeedForward(testData.Input)
			network.BackPropagation(testData.Output)
		}
	}
}

func Benchmark441(b *testing.B) {
	network := nn.New(4, 4, 1)
	for n := 0; n < b.N; n++ {
		for _, testData := range allTestData {
			network.FeedForward(testData.Input)
			network.BackPropagation(testData.Output)
		}
	}
}
