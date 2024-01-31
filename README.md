# iui_kdtree

A generic k-d tree-powered k-NN classifier written in C++20. While this is a university assignment,
it's decently optimized and flexible compared to similar libraries I found on GitHub.

I plan to use this in a future project, so further enhancements and documentation may be coming in the future.

At the moment, more information can be found in raport.odt (available only in Polish)

## k-NN classifier example

A data point is expressed with any aggregate type that may be decomposed into compatible values
using structured bindings:

```c++
struct DataPoint {
    std::array<float, 3> position;
    int label;
};
```

In order to construct a classifier, you now only need a range of `DataPoint`s:

```c++
std::vector<DataPoint> dataPoints;
// ...fill dataPoints...
iui::KNNClassifier<iui::EuclideanDistanceMetric, int, float, 3> classifier(dataPoints);
```

You may then call `predict` on the classifier:
```c++
int predictedLabel = classifier.predict({0.3f, 0.1f, 0.1f});
```

## Dimensionality reduction

The classifier may optionally take a dimensionality reducer type as a template parameter. For example, you may write:
```c++
iui::KNNClassifier<iui::EuclideanDistanceMetric, int, float, 3, iui::PrincipalComponentAnalysis, 2> classifier(dataPoints);
```
which will internally reduce the dataset from 3 to 2 dimensions using PCA.
`iui::PrincipalComponentAnalysis` is defined in `pca.hpp` and requires [Eigen](https://github.com/libigl/eigen).

Dimensionality reduction is entirely internal to the classifier and is transparent to the user.

```python
def createNode(int firstEntry, int lastEntry) -> Node:
	if lastEntry - firstEntry < leafThreshold
		return Leaf(firstEntry, lastEntry)

	split := findSplit(firstEntry, lastEntry)
	middleEntry := partition(firstEntry, lastEntry, split)

	return InnerNode(
		split = split,
		leftChild = createNode(firstEntry, middleEntry)
		rightChild = createNode(middleEntry, lastEntry)
	)

rootNode := createNode(0, entries.length())
```