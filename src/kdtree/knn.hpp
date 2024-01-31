
#ifndef KNN_HPP
#define KNN_HPP


#include "kdtree.hpp"
#include "metrics.hpp"

#include <queue>
#include <map>

namespace iui {

	namespace detail {

		template<typename TLabel>
		struct LabelScore {
			int frequency = 0;
			double negTotalDistance = 0.0;
			TLabel label {};

			friend auto operator<=>(const LabelScore& lhs, const LabelScore& rhs) {
				return std::make_tuple(lhs.frequency, rhs.negTotalDistance) <=> std::make_tuple(rhs.frequency, rhs.negTotalDistance);
			}
		};

	}

	template<typename TCoord, int NDimsSrc, int NDimsDst>
	struct NoDimensionalityReduction {
		static_assert(NDimsSrc == NDimsDst);

		static constexpr int NumInputDims = NDimsSrc;
		static constexpr int NumOutputDims = NDimsDst;

		using InputType = Vec<TCoord, NumInputDims>;
		using OutputType = Vec<TCoord, NumOutputDims>;

		explicit NoDimensionalityReduction(std::ranges::range auto range) {

		}

		auto reduce(const InputType& input) const {
			return input;
		}

	};

	inline double divOrZero(double a, double b) {
		if(b == 0) {
			return 0.0;
		}
		return a / b;
	}

	struct KNNClassifierStats {
		int totalPredictions = 0;
		int accuratePredictions = 0;
		int64_t pointsConsidered = 0;
		int64_t pointsSkipped = 0;

		[[nodiscard]] double accuracy() const {
			return divOrZero(accuratePredictions, totalPredictions);
		}

		[[nodiscard]] double efficiency() const {
			return divOrZero(pointsSkipped, pointsConsidered);
		}
	};



	template<
		typename TMetric,
		typename TLabel,
		typename TCoord,
		int NDims,
		template<typename, int, int> typename TDimensionalityReducer = NoDimensionalityReduction,
		int NTreeDims = NDims
	>
	struct KNNClassifier {

		static inline constexpr int NumTreeDimensions = NTreeDims;

		using MetricType = TMetric;
		using PointType = Vec<TCoord, NDims>;
		using DistanceType = double;
		using DimensionalityReducerType = std::conditional_t<
			NTreeDims < NDims,
			TDimensionalityReducer<TCoord, NDims, NTreeDims>,
			NoDimensionalityReduction<TCoord, NDims, NTreeDims>
		>;

		template<std::ranges::sized_range TRange>
			requires detail::AggregateEntryInitType<std::remove_cvref_t<std::ranges::range_value_t<TRange>>>
		explicit KNNClassifier(TRange&& range)
			: dimensionalityReducer_(range | std::views::transform([](auto&& entry) {
				const auto& [position, label] = entry;
		      	return position;
		      })),
			kdTree_((range | std::views::transform([this](auto&& entry) {
				const auto& [position, label] = entry;
				return typename decltype(kdTree_)::EntryType {dimensionalityReducer_.reduce(position), label};
			})), detail::KDTreeFromRangeTagT{})
		{

		}


		[[nodiscard]] TLabel predict(
			const PointType& point,
			int k = 3,
			std::optional<DistanceType> initialDist = std::nullopt,
			std::optional<TLabel> trueLabel = std::nullopt
		) {


			using TDistance = double;
			using TEntry = typename std::remove_cvref_t<decltype(kdTree_)>::EntryType;

			auto reducedPoint = dimensionalityReducer_.reduce(point);

			k = std::min<int>(k, kdTree_.numEntries());
			if(k < 1) {
				throw std::invalid_argument("k must be positive");
			}

			static constexpr double Epsilon = 1e-6;

			TDistance searchRadius = Epsilon;
			if(defaultSearchRadius < std::numeric_limits<double>::max()) {
				searchRadius = defaultSearchRadius;
			}
			if(initialDist) {
				searchRadius = *initialDist;
			}

			while(true) {
				if(std::isinf(searchRadius)) {
					throw std::runtime_error("cannot find any viable points (is the metric's predicate broken?)");
				}

				struct Candidate {
					DistanceType distance;
					TLabel label;
				};
				std::vector<Candidate> candidates;
				double totalDist = 0.0;
				int64_t entriesVisited = 0;

				kdTree_.walk(
					[&](const TEntry& entry) {
						entriesVisited++;
						auto distance = TMetric::distance(reducedPoint, entry.coord);
						totalDist += distance;
						if(distance < searchRadius) {
							candidates.push_back({distance, entry.label});
						}
					},
					[&](auto&& hbox) {
						return TMetric::intersectsSearchSpace(hbox, reducedPoint, searchRadius);
					}
				);
				auto averageDistance = divOrZero(totalDist, entriesVisited);

				if(candidates.size() < k) {
					searchRadius = std::max(searchRadius * 2.0, averageDistance);
					continue;
				}



				std::partial_sort(
					candidates.begin(),
					candidates.begin()+k,
					candidates.end(),
					[](const Candidate& a, const Candidate& b) {
						return a.distance < b.distance;
					}
				);
				candidates.resize(k);

				double maxCandDist = 0.0;
				for(int i=0; i<k; i++) {
					maxCandDist = std::max(maxCandDist, candidates[i].distance);
				}
				if(maxCandDist > Epsilon) {
					defaultSearchRadius = maxCandDist * 2.0;
				}

				std::map<TLabel, detail::LabelScore<TLabel>> labelScoresMap;
				for(auto& cand: candidates) {
					detail::LabelScore<TLabel>& score = labelScoresMap[cand.label];
					score.frequency += 1;
					score.negTotalDistance -= cand.distance;
					score.label = cand.label;
				}
				auto bestScore = std::max_element(
					labelScoresMap.begin(),
					labelScoresMap.end(),
					labelScoresMap.value_comp()
				);

				TLabel result = bestScore->second.label;

				stats.pointsConsidered += kdTree_.numEntries();
				stats.pointsSkipped += kdTree_.numEntries() - entriesVisited;

				if(trueLabel.has_value()) {
					stats.totalPredictions += 1;
					stats.accuratePredictions += result == trueLabel.value();
				}

				return result;
			}

		}

		[[nodiscard]] const KNNClassifierStats& getStats() const {
			return stats;
		}

		void resetStats() {
			stats = {};
		}

	private:
		double defaultSearchRadius = std::numeric_limits<double>::max();
		KNNClassifierStats stats;
		DimensionalityReducerType dimensionalityReducer_;
		KDTree<TLabel, NTreeDims, TCoord> kdTree_;
	};


}

#endif //KNN_HPP
