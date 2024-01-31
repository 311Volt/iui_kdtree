
#ifndef PCA_HPP
#define PCA_HPP

#include "Vec.hpp"
#include <eigen3/Eigen/SVD>

namespace iui {

	template<typename TCoord, int NDimsSrc, int NDimsDst>
	class PrincipalComponentAnalysis {
	public:
		static constexpr int NumInputDims = NDimsSrc;
		static constexpr int NumOutputDims = NDimsDst;

		using InputType = Vec<TCoord, NumInputDims>;
		using OutputType = Vec<TCoord, NumOutputDims>;


		explicit PrincipalComponentAnalysis(std::ranges::sized_range auto range) {

			std::vector<InputType> observations(std::ranges::begin(range), std::ranges::end(range));

			Eigen::MatrixXf M(observations.size(), NumInputDims);
			for(int i=0; i<observations.size(); i++) {
				observations[i].forEachEnumerated([&](int axis, auto&& val) {
					M(i, axis) = val;
				});
			}
			Eigen::BDCSVD<Eigen::MatrixXf> svd(M, Eigen::ComputeFullV);

			pcaTransform = svd.matrixV().leftCols(NumOutputDims);
		}

		auto reduce(const InputType& input) const {

			Eigen::RowVectorXf inputVec(NumInputDims);
			input.forEachEnumerated([&](int i, auto&& v) {
				inputVec(i) = v;
			});

			Eigen::RowVectorXf outputVec = inputVec * pcaTransform;
			OutputType output;
			output.forEachEnumerated([&](int i, TCoord& v) {
				v = outputVec(i);
			});
			return output;
		}


	private:
		Eigen::MatrixXf pcaTransform;
	};

}


// #include "Vec.hpp"
// #include <armadillo>
// #include <ranges>
//
// namespace iui {
//
// 	template<typename TCoord, int NDimsSrc, int NDimsDst>
// 	struct PrincipalComponentAnalysis {
//
// 		static constexpr int NumInputDims = NDimsSrc;
// 		static constexpr int NumOutputDims = NDimsDst;
//
// 		using InputType = Vec<TCoord, NumInputDims>;
// 		using OutputType = Vec<TCoord, NumOutputDims>;
//
// 		explicit PrincipalComponentAnalysis(std::ranges::sized_range auto range) {
// 			std::vector<InputType> observations(std::ranges::begin(range), std::ranges::end(range));
//
// 			arma::Mat<float> inputs(observations.size(), NumInputDims);
// 			for(int i=0; i<observations.size(); i++) {
// 				observations[i].forEachEnumerated([&](int axis, auto&& val) {
// 					inputs(i, axis) = val;
// 				});
// 			}
//
// 			arma::princomp(W, inputs);
// 			W = W(arma::span::all, arma::span(0, NumOutputDims-1));
// 			//W [in, out]
// 		}
//
// 		auto reduce(const InputType& input) const {
// 			arma::Row<float> X(NumInputDims);
// 			input.forEachEnumerated([&](int axis, auto&& val) {
// 				X[axis] = val;
// 			});
//
// 			arma::Row<float> R = X * W;
//
// 			OutputType output;
// 			output.forEachEnumerated([&](int axis, typename OutputType::ElementType& val) {
// 				val = R[axis];
// 			});
//
// 			return output;
// 		}
// 	private:
// 		arma::Mat<float> W;
// 	};
//
// }

#endif //PCA_HPP
