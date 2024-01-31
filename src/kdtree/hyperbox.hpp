
#ifndef HYPERBOX_HPP
#define HYPERBOX_HPP


#include <array>
#include <ranges>
#include <algorithm>
#include <format>
#include <sstream>

#include "Vec.hpp"

namespace iui {
	inline constexpr size_t CacheLineSize = 64;

	template<std::ranges::sized_range TRange>
	auto RangeToVector(TRange&& range) {
		using ValueT = std::remove_cvref_t<std::ranges::range_value_t<TRange>>;
		std::vector<ValueT> result;
		result.reserve(std::ranges::size(range));
		for(auto&& i: range) {
			result.emplace_back(i);
		}
		return result;
	}

	namespace detail {
		inline void checkAxis(int axis, int nDims) {
			if(axis < 0 || axis >= nDims) {
				throw std::domain_error(
					std::format("cannot split axis {} of a {}-dimensional hyperbox", axis, nDims)
				);
			}
		}
	}

	template<typename TCoord, int NDims>
	class Hyperbox {
	public:
		static inline constexpr int NumDimensions = NDims;
		using CoordType = TCoord;
		using PointType = Vec<TCoord, NDims>;
		PointType pos0, pos1;

		[[nodiscard]] bool contains(const PointType& point) const {
			for(int i=0; i<NDims; i++) {
				if(point[i] < pos0[i] || point[i] > pos1[i]) {
					return false;
				}
			}
			return true;
		}

		[[nodiscard]] bool overlaps(const Hyperbox& other) const {
			for(int i=0; i<NDims; i++) {
				if(other.pos0[i] > pos1[i] || other.pos1[i] < pos0[i]) {
					return false;
				}
			}
			return true;
		}

		struct Split {
			int axis;
			TCoord value;
		};

		[[nodiscard]] std::pair<Hyperbox, Hyperbox> split(Split split) const {
			detail::checkAxis(split.axis, NDims);
			split.value = std::clamp(split.value, pos0[split.axis], pos1[split.axis]);
			Hyperbox s1 = (*this);
			Hyperbox s2 = (*this);
			s1.pos1[split.axis] = split.value;
			s2.pos0[split.axis] = split.value;
			return {s1, s2};
		}

		template<std::ranges::sized_range TRange>
			requires std::is_convertible_v<std::ranges::range_value_t<TRange>, PointType>
		[[nodiscard]] static Hyperbox of(TRange&& items) {
			PointType pos0, pos1;
			pos0.fill(std::numeric_limits<TCoord>::max());
			pos1.fill(std::numeric_limits<TCoord>::min());
			for(const auto& item: items) {
				for(int i=0; i<NDims; i++) {
					pos0[i] = std::min(pos0[i], item[i]);
					pos1[i] = std::max(pos1[i], item[i]);
				}
			}
			return Hyperbox {pos0, pos1};
		}

		void print(const std::string& fmt = "{:.2f}", std::ostream& os = std::cout) const {
			os << "[";
			for(int i=0; i<NDims; i++) {
				auto fmtFull = fmt + "-" + fmt;
				os	<< std::vformat(fmtFull, std::make_format_args((double)pos0[i], (double)pos1[i]))
					<< ((i<NDims-1) ? "," : "");
			}
			os << "]";
		}

		[[nodiscard]] std::string toString(const std::string& fmt = "{:2f}") const {
			std::stringstream ss;
			print(fmt, ss);
			return ss.str();
		}


	private:
		template<PointType Hyperbox::*ptr>
		class BasicScopedSplitter {
		public:
			BasicScopedSplitter(Hyperbox& hyperbox, Split split): axis(split.axis), hyperbox(hyperbox) {
				split.value = std::clamp(split.value, hyperbox.pos0[split.axis], hyperbox.pos1[split.axis]);
				originalValue = (hyperbox.*ptr)[split.axis];
				(hyperbox.*ptr)[axis] = split.value;
			}
			~BasicScopedSplitter() {
				(hyperbox.*ptr)[axis] = originalValue;
			}
		private:
			int axis;
			TCoord originalValue;
			Hyperbox& hyperbox;
		};
	public:
		using ScopedLeftSplitter = BasicScopedSplitter<&Hyperbox::pos1>;
		using ScopedRightSplitter = BasicScopedSplitter<&Hyperbox::pos0>;
	};


}

#endif //HYPERBOX_HPP
