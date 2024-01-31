#include <iostream>
#include <random>
#include <chrono>

#include "mnistReader.hpp"
#include "dryBeansReader.hpp"
#include "kdtree/knn.hpp"
#include "kdtree/pca.hpp"

template<typename TClassifier>
void benchmarkClassifier(auto trainingData, auto valData, int k = 3, std::optional<double> initRadius = {}) {

	auto t0 = std::chrono::high_resolution_clock::now();
	TClassifier classifier(trainingData);
	auto t1 = std::chrono::high_resolution_clock::now();

	auto t2 = std::chrono::high_resolution_clock::now();
	for(const auto& valSmp: valData) {
		const auto& [pos, label] = valSmp;
		auto prediction = classifier.predict(pos, k, initRadius, label);
	}
	auto t3 = std::chrono::high_resolution_clock::now();

	using DurMillis = std::chrono::duration<double, std::milli>;
	double millisCtor = std::chrono::duration_cast<DurMillis>(t1 - t0).count();
	double millisTest = std::chrono::duration_cast<DurMillis>(t3 - t2).count();

	std::cout << std::format(
		"n={:3d}, k={:2d}, accuracy: {:.2f}% ({} / {}), efficiency: {:.2f}%, ctor {:.2f} ms, test {:.2f} ms\n",
		TClassifier::NumTreeDimensions,
		k,
		100.0 * classifier.getStats().accuracy(),
		classifier.getStats().accuratePredictions,
		classifier.getStats().totalPredictions,
		100.0 * classifier.getStats().efficiency(),
		millisCtor,
		millisTest);
}

void testCurseOfDimensionality(auto&& mnistTrain, auto&& mnistVal) {

	printf("testing curse of dimensionality...\n");
	[&]<std::size_t... Idxs>(std::index_sequence<Idxs...> _){
		(([&](auto nDims) {
			benchmarkClassifier<
			iui::KNNClassifier<
				iui::ManhattanDistanceMetric, int, int, 784, iui::PrincipalComponentAnalysis, nDims()
			>>(mnistTrain, mnistVal, 3);
		})(std::integral_constant<int, Idxs + 1>{}), ...);
	}(std::make_index_sequence<50>{});
}

void benchmarkManhattan(auto&& mnistTrain, auto&& mnistVal) {
	printf("benchmarking MNIST dataset (Manhattan)...\n");
	[&]<std::size_t... Idxs>(std::index_sequence<Idxs...> _){
		(([&](auto nDims) {
			for(int k=1; k<=5; k+=2) {
				benchmarkClassifier<
				iui::KNNClassifier<
					iui::ManhattanDistanceMetric, int, int, 784, iui::PrincipalComponentAnalysis, nDims()
				>>(mnistTrain, mnistVal, k);
			}
		})(std::integral_constant<int, Idxs>{}), ...);
	}(std::index_sequence<3, 8, 16, 72, 784>{});
}


void benchmarkEuclidean(auto&& mnistTrain, auto&& mnistVal) {
	printf("benchmarking MNIST dataset (Euclidean)...\n");
	[&]<std::size_t... Idxs>(std::index_sequence<Idxs...> _){
		(([&](auto nDims) {
			for(int k=1; k<=5; k+=2) {
				benchmarkClassifier<
				iui::KNNClassifier<
					iui::EuclideanDistanceMetric, int, int, 784, iui::PrincipalComponentAnalysis, nDims()
				>>(mnistTrain, mnistVal, k);
			}
		})(std::integral_constant<int, Idxs>{}), ...);
	}(std::index_sequence<3, 8, 16, 72, 784>{});
}

void benchmarkPaletteQuantization() {

	struct PaletteColor {
		iui::Vec3f position;
		int value;
	};
	std::vector<PaletteColor> trainingSet, validationSet;

	auto randv3f = [&]() {
		static std::minstd_rand0 random {std::random_device{}()};
		static auto dist = std::uniform_real_distribution(0.0, 1.0);
		return iui::Vec3f(dist(random), dist(random), dist(random));
	};


	for(int i=0; i<65536; i++) {
		trainingSet.push_back(PaletteColor {
			.position = randv3f(),
			.value = i
		});
		validationSet.push_back(PaletteColor {
			.position = randv3f(),
			.value = 0
		});
	}

	using TClassifier = iui::KNNClassifier<iui::EuclideanDistanceMetric, int, float, 3>;
	benchmarkClassifier<TClassifier>(trainingSet, validationSet, 1);
}

void simpleUsageExample(auto&& mnistTrain, auto&& mnistVal) {
	iui::KNNClassifier<iui::EuclideanDistanceMetric, int, int, 784, iui::PrincipalComponentAnalysis, 12> classifier(mnistTrain);

	std::minstd_rand0 random {std::random_device{}()};
	std::vector<NumberMNIST> sample;
	std::ranges::sample(mnistVal, std::back_inserter(sample), 10, random);

	for(const auto& digit: sample) {
		auto prediction = classifier.predict(digit.image, 3, {}, digit.label);
		digit.print();
		std::cout << std::format("label: {}, prediction: {} ({})\n", digit.label, prediction, digit.label==prediction ? "ok" : "err");
	}
}

int main()
{
	constexpr int bruh = sizeof(typename iui::KDTree<int, 3, float>::Node);
	printf("reading MNIST dataset...\n");
	auto mnistTrain = readSampleFileMNIST("trainingsample.csv");
	auto mnistVal = readSampleFileMNIST("validationsample.csv");

	simpleUsageExample(mnistTrain, mnistVal);

	benchmarkPaletteQuantization();

	benchmarkEuclidean(mnistTrain, mnistVal);
	benchmarkManhattan(mnistTrain, mnistVal);

	return 0;
}
