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
	for k := 0; k < 1000; k++ {
		seed := time.Now().UTC().UnixNano()
		rand.Seed(seed)
		log.Printf("seed %d ", seed)
		n := nn.New(3, 3, 1)

		generation := 0
		for n.Error > 0.001 {
			for _, d := range data {
				n.FeedForward([]float64{d.A, d.B,
					//some noise extremely speedups education
					rand.Float64()})
				n.BackPropagation([]float64{d.C})
			}
			generation++
			//log.Printf("Generation %d : error= %f", generation, n.Error)
		}

		log.Printf("Generation %d : error= %f", generation, n.Error)
		for _, d := range data {
			n.FeedForward([]float64{d.A, d.B})
			result := n.GetResults()[0]
			log.Printf("%d XOR %d : %d = %d", int(d.A), int(d.B), int(d.C), int(result+0.5))
		}
	}
}
