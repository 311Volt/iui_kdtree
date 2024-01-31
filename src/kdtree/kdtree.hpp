
#ifndef KDTREE_HPP
#define KDTREE_HPP

#include <array>
#include <ranges>
#include <algorithm>
#include <deque>
#include <random>
#include <format>
#include <memory>

#include "Vec.hpp"
#include "hyperbox.hpp"

namespace iui {

	inline bool TreeDebug = false;

	template<typename... Ts>
	struct overloaded: public Ts... {
		using Ts::operator()...;
	};

	namespace detail {
		template<typename T>
		concept AggregateEntryInitType =
			   std::is_aggregate_v<std::remove_cvref_t<T>>
			&& std::is_class_v<std::remove_cvref_t<T>>;

		struct KDTreeFromRangeTagT{};
	};

	enum class BalancingPolicy {
		Exact,
		Approximate
	};

	template<typename TLabel, int NDims, typename TCoord = double>
	class KDTree {
	public:


		using IndexType = Vec<TCoord, NDims>;
		using ElementType = TLabel;
		using HyperboxType = Hyperbox<TCoord, NDims>;
		using HyperboxSplitType = typename HyperboxType::Split;


		struct EntryType {
			IndexType coord;
			ElementType label;
		};


		struct Node {
			struct InnerNode {
				HyperboxSplitType split;
				Node* lchild;
				Node* rchild;
			};

			Node* parent;
			std::variant<std::span<const EntryType>, InnerNode> data;
		};

		using InnerNodeType = typename Node::InnerNode;

		static constexpr size_t MaxDepth = 64;
		static constexpr size_t MaxLeafElements = std::max<size_t>(2, 2 * CacheLineSize / sizeof(IndexType));


		template<std::ranges::sized_range TRange>
			requires (std::is_convertible_v<std::ranges::range_value_t<TRange>, EntryType>)
		explicit KDTree(TRange&& items, detail::KDTreeFromRangeTagT _ = {}) {
			entries_ = RangeToVector(items);
			rootHyperbox_ = HyperboxType::of(entries_ | std::views::transform([](const EntryType& e){return e.coord;}));
			rootNode_ = createNode(entries_, nullptr);
			//printf("tree has %d kB\n", (sizeof(Node) * nodes_.size()) / 1024);
		}

		template<std::ranges::sized_range TRange>
			requires detail::AggregateEntryInitType<std::remove_cvref_t<std::ranges::range_value_t<TRange>>>
		explicit KDTree(TRange&& items) : KDTree(std::views::transform(items, [](auto&& item) {
			const auto& [position, value] = item;
			return EntryType {position, value};
		}), detail::KDTreeFromRangeTagT {}) {}

		[[nodiscard]] const Node* rootNode() const {
			return rootNode_;
		}

		KDTree(const KDTree&) = delete;
		KDTree& operator=(const KDTree&) = delete;
		KDTree(KDTree&&) = default;
		KDTree& operator=(KDTree&&) = default;

		template<std::invocable<EntryType> FnT, std::invocable<HyperboxType> PredFnT>
		void walk(FnT&& fn, PredFnT&& hboxPredicate) const {
			HyperboxType hbox = rootHyperbox_;
			struct DataVisitor {
				FnT fn;
				PredFnT hboxPredicate;
				HyperboxType& hbox;

				void operator()(const std::span<const EntryType> entries) {
					for(const auto& entry: entries) {
						fn(entry);
					}
				}
				void operator()(InnerNodeType innerNode) {
					{
						typename HyperboxType::ScopedLeftSplitter splitter(hbox, innerNode.split);
						if(hboxPredicate(hbox)) {
							std::visit(*this, innerNode.lchild->data);
						}
					}
					{
						typename HyperboxType::ScopedRightSplitter splitter(hbox, innerNode.split);
						if(hboxPredicate(hbox)) {
							std::visit(*this, innerNode.rchild->data);
						}
					}

				}
			};
			return std::visit(DataVisitor{fn, hboxPredicate, hbox}, rootNode()->data);
		}

		[[nodiscard]] size_t numEntries() const {
			return entries_.size();
		}

	private:

		struct HyperboxSplitRecord {
			double score {};
			HyperboxSplitType split {};
		};

		[[nodiscard]] static HyperboxSplitRecord trySplit(std::span<EntryType> entries, int axis) {
			int mid = entries.size() / 2;
			std::nth_element(entries.begin(), entries.begin()+mid, entries.end(), [axis](const EntryType& a, const EntryType& b) {
				return a.coord[axis] < b.coord[axis];
			});
			auto median = entries[mid].coord[axis];
			auto midpoint = std::partition(entries.begin(), entries.end(), [axis, median](const EntryType& entry) {
				return entry.coord[axis] < median;
			});
			auto leftSubspan = std::span {entries.begin(), midpoint};
			auto rightSubspan = std::span {midpoint, entries.end()};

			int sizeDiff = std::abs(std::ssize(leftSubspan) - std::ssize(rightSubspan));
			int maxAbsInvScore = std::ssize(entries) - (std::ssize(entries) % 2);
			int absInvScore = std::ssize(entries) - sizeDiff;

			double score = double(absInvScore) / double(maxAbsInvScore);

			return HyperboxSplitRecord {
				.score = score,
				.split = {
					.axis = axis,
					.value = median
				}
			};
		}

		std::array<std::span<EntryType>, 2> applySplit(std::span<EntryType> entries, HyperboxSplitType split) {
			auto belongsToLeft = [&split](const EntryType& entry) {
				return entry.coord[split->axis] < split->value;
			};
			auto partition = std::partition(entries.begin(), entries.end(), belongsToLeft);

			auto lChildEntries = std::span(entries.begin(), partition);
			auto rChildEntries = std::span(partition, entries.end());
			return {lChildEntries, rChildEntries};
		}

		// [[nodiscard]] static std::optional<HyperboxSplitType> pickSplit(std::span<HyperboxSplitRecord> splits) {
		// 	std::ranges::sort(splits, [](const HyperboxSplitRecord& a, const HyperboxSplitRecord& b) {
		// 		return a.score > b.score;
		// 	});
		//
		// 	if(splits[0].score == 0.0) {
		// 		return std::nullopt;
		// 	}
		// 	auto first_viable = splits.begin();
		// 	auto last_viable = std::find_if_not(splits.begin(), splits.end(), [&](auto&& a) {
		// 		return a.score >= ViableScoreThreshold;
		// 	});
		//
		// 	if(std::distance(first_viable, last_viable) > 0) {
		// 		static std::mt19937_64 gen(std::random_device{}());
		// 		HyperboxSplitRecord selection;
		// 		std::sample(first_viable, last_viable, &selection, 1, gen);
		// 		return selection.split;
		// 	} else {
		// 		return splits[0].split;
		// 	}
		// }

		[[nodiscard]] static std::optional<HyperboxSplitType> findApproximateSplit(std::span<EntryType> entries) {
			std::vector<HyperboxSplitRecord> splits;

			static std::mt19937_64 gen(std::random_device{}());
			static constexpr double ViableScoreThreshold = 0.9;

			for(int i=0; i<std::min<int>(NDims, 2.0 + 2.0 * std::log2(NDims)); i++) {
				int axis = std::uniform_int_distribution<int>(0, NDims-1)(gen);
				auto rec = trySplit(entries, axis);
				if(rec.score > ViableScoreThreshold) {
					return rec.split;
				}
				splits.push_back(rec);
			}

			auto best = std::max_element(splits.begin(), splits.end(), [](auto&& a, auto&& b) {
				return a.score > b.score;
			});
			if(best == splits.end() || best->score == 0.0) {
				return std::nullopt;
			}
			return best->split;
		}





		Node* createNode(std::span<EntryType> entries, Node* parent) {

			nodes_.push_back({});
			Node& result = nodes_.back();

			result.parent = parent;
			//result.hyperbox = hbox;
			if(entries.size() <= MaxLeafElements) {
				result.data = entries;
			} else {

				auto split = findApproximateSplit(entries);

				if(not split) {
					result.data = entries;
				} else {
					auto partition = std::partition(entries.begin(), entries.end(), [&split](const EntryType& entry) {
						return entry.coord[split->axis] < split->value;
					});

					//auto [lHbox, rHbox] = hbox.split(*split);

					auto lChildEntries = std::span(entries.begin(), partition);
					auto rChildEntries = std::span(partition, entries.end());

					if(TreeDebug) {
						for(Node* it = &result; it; it = it->parent) {
							printf(" ");
						}
						// if(NDims <= 4) {
						// 	result.hyperbox.print();
						// }
						printf("[%d | %.2f] %d -> %d / %d\n",
							split->axis,
							(double)split->value,
							(int)entries.size(),
							(int)lChildEntries.size(),
							(int)rChildEntries.size());

					}
					// result.hyperbox.print();

					result.data = InnerNodeType {
						.split = *split,
						.lchild = createNode(lChildEntries, &result),
						.rchild = createNode(rChildEntries, &result)
					};
				}
			}

			return &result;
		}

		HyperboxType rootHyperbox_;
		std::deque<Node> nodes_;
		std::vector<EntryType> entries_;
		Node* rootNode_ {};
	};

}

#endif //KDTREE_HPP

