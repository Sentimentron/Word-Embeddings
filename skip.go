package main

import (
	"fmt"
	mlp "github.com/r9y9/nnet/mlp"
	"io/ioutil"
	"math"
	"math/rand"
	"sort"
	"strings"
)

const (
	NumDimensions = 10
	WindowSize    = 5
)

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func DSigmoid(x float64) float64 {
	return x * (1.0 - x)
}

func Tanh(x float64) float64 {
	return math.Tanh(x)
}

func DTanh(x float64) float64 {
	return 1.0 - math.Pow(x, 2)
}

func Linear(x float64) float64 {
	return x
}

func DLinear(x float64) float64 {
	return 1.0
}

func HardTanh(x float64) float64 {
	if x < -1.0 {
		return -1.0
	} else if x > 1.0 {
		return 1.0
	} else {
		return x
	}
}

func DHardTanh(x float64) float64 {
	if x < -1.0 {
		return 0.0
	} else if x > 1.0 {
		return 0.0
	} else {
		return 1.0
	}
}

func createNetwork() *mlp.MLP {
	d := mlp.NewMLP()
	h := NumDimensions * WindowSize
	// Add Linear input layer
	d.ConnectLayer(mlp.NewHiddenLayer(h, h, Linear, DLinear))
	// Add HardTanh layer
	d.ConnectLayer(mlp.NewHiddenLayer(h, h, HardTanh, DHardTanh))
	// Final linear layer
	d.ConnectLayer(mlp.NewHiddenLayer(h, 1, Linear, DLinear))
	return d
}

func computeWordDist(d1, d2 []float64) float64 {
	sum := 0.0
	for i, d := range d1 {
		sum += (d - d2[i]) * (d - d2[i])
	}
	return sum
}

type wordDistance struct {
	word string
	dist float64
}

type wordDistanceVec []wordDistance

func (ws wordDistanceVec) Len() int           { return len(ws) }
func (ws wordDistanceVec) Swap(i, j int)      { ws[i], ws[j] = ws[j], ws[i] }
func (ws wordDistanceVec) Less(i, j int) bool { return ws[i].dist < ws[j].dist }

func mostSimilarTo(word string, int2Word []string, word2Int map[string]int, wem [][]float64) {
	wordId := word2Int[word]
	wordVec := wem[wordId]
	fmt.Printf("%s ", word)
	wsVec := make([]wordDistance, len(word2Int))
	for i, w := range wem {
		if i == wordId {
			continue
		}
		dist := computeWordDist(wordVec, w)
		wsVec[i] = wordDistance{
			int2Word[i],
			dist,
		}
	}
	sort.Sort(wordDistanceVec(wsVec))
	for _, d := range wsVec[:5] {
		fmt.Printf("%s (%.2f) ", d.word, d.dist)
	}
	fmt.Println()
}

func main() {

	// Create a randomness source
	rnd := rand.New(rand.NewSource(8881182))
	// Read input
	in, err := ioutil.ReadFile("news.txt")
	if err != nil {
		panic(err)
	}

	// Split into lines
	lines := strings.Split(string(in), "\n")
	// Split into documents
	docs := make([][]string, len(lines))
	for i, l := range lines {
		docs[i] = strings.Split(l, " ")
	}
	// Count the number of distinct words
	wordCounts := make(map[string]int)
	for _, l := range docs {
		for _, w := range l {
			wordCounts[w]++
		}
	}

	// Word enumeration
	wordCount := 0
	int2Word := make([]string, wordCount)
	word2Int := make(map[string]int)
	for w := range wordCounts {
		int2Word = append(int2Word, w)
		word2Int[w] = wordCount
		wordCount++
	}

	// Allocate a matrix of the right size
	wem := make([][]float64, wordCount)
	// Initialise WEM
	for i := 0; i < wordCount; i++ {
		wem[i] = make([]float64, NumDimensions)
		for j := 0; j < NumDimensions; j++ {
			wem[i][j] = rnd.NormFloat64() * 0.01
		}
	}
	tmpCorrect := make([][]float64, WindowSize)
	tmpWrong := make([][]float64, WindowSize)
	tmpIdCorrect := make([]int, WindowSize)

	// Print the initial similarities
	mostSimilarTo("burger", int2Word, word2Int, wem)

	// Do the skip-gram training
	for prog, l := range docs {
		fmt.Println(prog, len(docs))
		for i := 0; i < len(l)-WindowSize; i++ {
			for j := 0; j < WindowSize; j++ {
				// Find the word
				offset := i + j
				word := l[offset]
				wordId := word2Int[word]
				// Copy the word's weights into the temporary matrix
				tmpCorrect[j] = wem[wordId]
				tmpWrong[j] = wem[wordId]
				tmpIdCorrect[j] = wordId
			}
			// Mutate a word
			tmpWrongOffset := rnd.Intn(WindowSize)
			for {
				if tmpWrongOffset != WindowSize/2 {
					break
				}
				tmpWrongOffset = rnd.Intn(WindowSize)
			}
			wrongWordId := rnd.Intn(len(int2Word))
			tmpWrong[tmpWrongOffset] = wem[wrongWordId]
			// Run training
			net := createNetwork()
			option := mlp.TrainingOption{
				LearningRate:       0.0001,
				Epoches:            3000,
				MiniBatchSize:      2,
				L2Regularization:   true,
				RegularizationRate: 1.0e-7,
				Monitoring:         false,
			}

			inputLayer1 := make([]float64, WindowSize*NumDimensions)
			inputLayer2 := make([]float64, WindowSize*NumDimensions)
			copyNetLayers(tmpCorrect, inputLayer1)
			copyNetLayers(tmpWrong, inputLayer2)
			output := make([][]float64, 2)
			output[0] = []float64{1}
			output[1] = []float64{0.99}
			err := net.Train([][]float64{inputLayer1, inputLayer2}, output, option)
			if err != nil {
				panic(err)
			}
			// Propagate weights from Linear layer to WEM
			inputHiddenLayer := net.HiddenLayers[0]
			weights := inputHiddenLayer.W
			biases := inputHiddenLayer.B
			newWeights := make([]float64, NumDimensions*WindowSize)
			dontCopy := false
			for i, ws := range weights {
				sum := 0.0
				for _, w := range ws {
					sum += w * inputLayer1[i]
				}
				sum -= biases[i]
				if math.IsNaN(sum) {
					dontCopy = true
				}
				newWeights[i] = sum
			}
			if !dontCopy {
				copyNetLayersBack(tmpCorrect, newWeights)
			}
		}
	}

	// Print the resulting similarities
	mostSimilarTo("burger", int2Word, word2Int, wem)
}

func copyNetLayers(in [][]float64, out []float64) {
	for i, m := range in {
		copy(out[len(m)*i:len(m)*i+i], m)
	}
}

func copyNetLayersBack(out [][]float64, in []float64) {
	for i, m := range out {
		copy(m, in[len(m)*i:len(m)*i+i])
	}
}
