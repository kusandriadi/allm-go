package allm

import "math"

// Float is a constraint for floating-point types used in vector operations.
type Float interface {
	~float32 | ~float64
}

// CosineSimilarity computes the cosine similarity between two vectors.
// Returns 0 if the vectors have different lengths or either is a zero vector.
func CosineSimilarity[T Float](a, b []T) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// EuclideanDistance computes the Euclidean distance between two vectors.
// Returns math.MaxFloat64 if the vectors have different lengths.
func EuclideanDistance[T Float](a, b []T) float64 {
	if len(a) != len(b) {
		return math.MaxFloat64
	}

	var sum float64
	for i := range a {
		diff := float64(a[i]) - float64(b[i])
		sum += diff * diff
	}

	return math.Sqrt(sum)
}

// DotProduct computes the dot product of two vectors.
// Returns 0 if the vectors have different lengths.
func DotProduct[T Float](a, b []T) float64 {
	if len(a) != len(b) {
		return 0
	}

	var result float64
	for i := range a {
		result += float64(a[i]) * float64(b[i])
	}

	return result
}
