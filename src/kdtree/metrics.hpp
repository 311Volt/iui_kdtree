
#ifndef METRICS_HPP
#define METRICS_HPP

#include "hyperbox.hpp"
#include <cmath>

namespace iui {

	template<typename T>
	concept DistanceMetric = requires
	{
		{T::distance(Vec3f{}, Vec3f{})};
		{T::intersectsSearchSpace(Hyperbox<float, 3>{}, Vec3f{}, 0.0)};
	};

	namespace detail {

		template<int N, typename T>
		inline T constPow(T x) {
			if constexpr(N == 1) {
				return x;
			} else if constexpr(N == 2) {
				return x*x;
			} else if constexpr(N == 3) {
				return x*x*x;
			} else if constexpr(N == 4) {
				return (x*x)*(x*x);
			} else {
				return std::pow(x, N);
			}
		}

		template<int N, typename T>
		inline T constRoot(T x) {
			if constexpr(N == 1) {
				return x;
			} else if constexpr(N == 2) {
				return std::sqrt(x);
			} else if constexpr(N == 3) {
				return std::cbrt(x);
			} else if constexpr(N == 4) {
				return std::sqrt(std::sqrt(x));
			} else {
				return std::pow(x, 1.0 / N);
			}
		}

		template<int N, typename T>
		inline T constAbsPow(T x) {
			if constexpr(N % 2 == 0) {
				return constPow<N>(x);
			} else {
				return std::fabs(constPow<N>(x));
			}
		}

	}


	template<int N>
	struct MinkowskiDistanceMetric {
		static_assert(N >= 1);

		template<typename TCoord, int NDims>
		[[nodiscard]] static double distance(const Vec<TCoord, NDims>& p1, const Vec<TCoord, NDims>& p2) {
			double result = 0.0;
			(p2 - p1).forEach([&](auto v) {
				result += detail::constAbsPow<N>(v);
			});
			return detail::constRoot<N>(result);
		}

		template<typename TCoord, int NDims>
		[[nodiscard]] static bool intersectsSearchSpace(const Hyperbox<TCoord, NDims>& hbox, const Vec<TCoord, NDims>& point, double maxDist) {
			double dist = detail::constAbsPow<N>(maxDist);
			for(int i=0; i<NDims; i++) {
				if(point[i] < hbox.pos0[i])
					dist -= detail::constAbsPow<N>(point[i] - hbox.pos0[i]);
				else if(point[i] > hbox.pos1[i])
					dist -= detail::constAbsPow<N>(point[i] - hbox.pos1[i]);
			}
			return dist >= 0;
		}
	};

	using ManhattanDistanceMetric = MinkowskiDistanceMetric<1>;
	using EuclideanDistanceMetric = MinkowskiDistanceMetric<2>;

}

#endif //METRICS_HPP
