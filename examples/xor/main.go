package main

import (
	"log"
	"math/rand"
	"time"

	"github.com/o5h/nn"
)

var data = []struct {
	A, B, C float64
}{
	{0, 0, 0},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 0},
}

func main() {
	seed := time.Now().UTC().UnixNano()
	rand.Seed(seed)
	n := nn.New(2, 3, 1)

	generation := 0
	for n.Error > 0.001 {
		for _, d := range data {
			n.FeedForward([]float64{d.A, d.B})
			n.BackPropagation([]float64{d.C})
		}
		generation++

	}

	log.Printf("seed %d : Generation %d : error= %f", seed, generation, n.Error)
	for _, d := range data {
		n.FeedForward([]float64{d.A, d.B})
		result := n.GetResults()[0]
		log.Printf(" Expected %f : actual = %f", d.C, result)
	}

}
