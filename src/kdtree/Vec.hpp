
#ifndef AXXEGRO_VEC_HPP
#define AXXEGRO_VEC_HPP

#include <array>
#include <concepts>
#include <functional>
#include <limits>
#include <cmath>
#include <utility>
#include <iostream>
#include <stdint.h>


#ifdef _MSC_VER
#define AXXEGRO_FORCE_INLINE [[msvc::forceinline]]
#else
#define AXXEGRO_FORCE_INLINE [[gnu::always_inline]]
#endif

namespace iui {


	template<typename T>
	concept Arithmetic = std::is_arithmetic_v<T>;

	template<Arithmetic From, Arithmetic To>
	constexpr bool IsNarrowingConversion = not requires(From x) {
		{To{x}};
	};

	namespace vec {

		template<typename T, int N>
		struct BaseNamedCoord {};

		struct Empty {};


//		template<typename T>
//		struct BaseNamedCoord<T, 0> {T x;};
//
//		template<typename T>
//		struct BaseNamedCoord<T, 1> {T y;};
//
//		template<typename T>
//		struct BaseNamedCoord<T, 2> {T z;};
//
//		template<typename T>
//		struct BaseNamedCoord<T, 3> {T w;};
//
//		/*
//		 * Provides the members x,y,z,w for vectors up to 4 elements.
//		 */
//		template<typename T, int N>
//		struct BaseNamedCoords :
//			public std::conditional_t<(N > 0), BaseNamedCoord<T, 0>, Empty>,
//			public std::conditional_t<(N > 1), BaseNamedCoord<T, 1>, Empty>,
//			public std::conditional_t<(N > 2), BaseNamedCoord<T, 2>, Empty>,
//			public std::conditional_t<(N > 3), BaseNamedCoord<T, 3>, Empty>
//		{
//			static_assert(N > 0 && N <= 4);
//			static constexpr int NumDimensions = N;
//			using ElementType = T;
//			using BNC_ID = std::true_type;
//		};

#define AXX_BNC_METADATA(type, ndim) \
	static constexpr int NumDimensions = ndim; \
	using ElementType = type; \
	using BNC_ID = std::true_type;

		template<typename T, int N>
		struct BaseNamedCoords {
			static_assert(N > 0 && N <= 4);
		};


		template<typename T>
		struct BaseNamedCoords<T, 1> {
			T x;
			AXX_BNC_METADATA(T, 1)
		};

		template<typename T>
		struct BaseNamedCoords<T, 2> {
			T x, y;
			AXX_BNC_METADATA(T, 2)
		};

		template<typename T>
		struct BaseNamedCoords<T, 3> {
			T x, y, z;
			AXX_BNC_METADATA(T, 3)
		};

		template<typename T>
		struct BaseNamedCoords<T, 4> {
			T x, y, z, w;
			AXX_BNC_METADATA(T, 4)
		};

#undef AXX_BNC_METADATA

		/*
		 * Provides the array data member for larger vectors.
		 */
		template<typename T, int N>
		struct BaseArrCoords {
			std::array<T, N> coord;
			static constexpr int NumDimensions = N;
			using ElementType = T;
			using BAC_ID = std::true_type;
		};

		template<typename T>
		concept InstOfBaseCoords = requires {
			typename T::ElementType;
			requires std::integral<decltype(T::NumDimensions)>;
		};

		template<typename T>
		concept InstOfBaseNamedCoords = InstOfBaseCoords<T> && std::same_as<typename T::BNC_ID, std::true_type>;

		template<typename T>
		concept InstOfBaseArrCoords = InstOfBaseCoords<T> && std::same_as<typename T::BAC_ID, std::true_type>;



		template<InstOfBaseNamedCoords T, int I>
		struct GetNamedCoordAt {
			using VT = std::remove_cvref_t<T>;
			AXXEGRO_FORCE_INLINE static constexpr auto from([[maybe_unused]] VT& v) -> typename T::ElementType& { return {}; }
			AXXEGRO_FORCE_INLINE static constexpr auto from([[maybe_unused]] const VT& v) -> typename T::ElementType const& { return {}; }
		};
		template<InstOfBaseNamedCoords T>
		struct GetNamedCoordAt<T, 0> {
			using VT = std::remove_cvref_t<T>;
			AXXEGRO_FORCE_INLINE static constexpr auto from(VT& v) -> typename T::ElementType& { return v.x; }
			AXXEGRO_FORCE_INLINE static constexpr auto from(const VT& v) -> typename T::ElementType const& { return v.x; }
			static constexpr size_t offset() {return offsetof(T, x);}
		};
		template<InstOfBaseNamedCoords T>
		struct GetNamedCoordAt<T, 1> {
			using VT = std::remove_cvref_t<T>;
			AXXEGRO_FORCE_INLINE static constexpr auto from(VT& v) -> typename T::ElementType& { return v.y; }
			AXXEGRO_FORCE_INLINE static constexpr auto from(const VT& v) -> typename T::ElementType const& { return v.y; }
			static constexpr size_t offset() {return offsetof(T, y);}
		};
		template<InstOfBaseNamedCoords T>
		struct GetNamedCoordAt<T, 2> {
			using VT = std::remove_cvref_t<T>;
			AXXEGRO_FORCE_INLINE static constexpr auto from(VT& v) -> typename T::ElementType& { return v.z; }
			AXXEGRO_FORCE_INLINE static constexpr auto from(const VT& v) -> typename T::ElementType const& { return v.z; }
			static constexpr size_t offset() {return offsetof(T, z);}
		};
		template<InstOfBaseNamedCoords T>
		struct GetNamedCoordAt<T, 3> {
			using VT = std::remove_cvref_t<T>;
			AXXEGRO_FORCE_INLINE static constexpr auto from(VT& v) -> typename T::ElementType& { return v.w; }
			AXXEGRO_FORCE_INLINE static constexpr auto from(const VT& v) -> typename T::ElementType const& { return v.w; }
			static constexpr size_t offset() {return offsetof(T, w);}
		};

		template<InstOfBaseNamedCoords Inst, int Begin = 0>
		constexpr bool ComputeCanUseFastSubscripting() {
			static_assert(Begin >= 0 && Begin <= Inst::NumElements);
			if constexpr(Begin == Inst::NumElements) {
				return true;
			} else {
				size_t expected = Begin * sizeof(typename Inst::ElementType);
				size_t actual = GetNamedCoordAt<Inst, Begin>::offset();
				return expected==actual && ComputeCanUseFastSubscripting<Inst, Begin+1>();
			}
		}

		template<InstOfBaseNamedCoords Inst>
		constexpr bool CanUseFastSubscripting = ComputeCanUseFastSubscripting<Inst>();


		template<typename T>
		concept InstOfBaseNamedCoordsWithFSS = InstOfBaseNamedCoords<T> && CanUseFastSubscripting<T>;

		template<typename T>
		concept InstOfBaseNamedCoordsWithoutFSS = InstOfBaseNamedCoords<T> && !CanUseFastSubscripting<T>;

		template<InstOfBaseArrCoords Inst>
		AXXEGRO_FORCE_INLINE inline auto At(Inst& v, int i) -> typename Inst::ElementType& {
			return v.coord[i];
		}

		template<InstOfBaseArrCoords Inst>
		AXXEGRO_FORCE_INLINE inline auto At(const Inst& v, int i) -> typename Inst::ElementType const& {
			return v.coord[i];
		}

		template<InstOfBaseNamedCoordsWithFSS Inst>
		AXXEGRO_FORCE_INLINE inline auto At(Inst& v, int i) -> typename Inst::ElementType& {
			return reinterpret_cast<typename Inst::ElementType*>(&v)[i];
		}

		template<InstOfBaseNamedCoordsWithFSS Inst>
		AXXEGRO_FORCE_INLINE inline auto At(const Inst& v, int i) -> typename Inst::ElementType const& {
			return reinterpret_cast<const typename Inst::ElementType*>(&v)[i];
		}

		template<InstOfBaseNamedCoordsWithoutFSS Inst, int Begin = 0>
		AXXEGRO_FORCE_INLINE inline auto At(Inst& v, int i) -> typename Inst::ElementType& {
			if constexpr (Begin < 0 || Begin >= Inst::NumDimensions) {
				return typename Inst::ElementType {};
			}
			if(i == Begin) {
				return GetNamedCoordAt<Inst, Begin>::from(v);
			} else {
				return At<Inst, Begin+1>(v, i);
			}
		}

		template<InstOfBaseNamedCoordsWithoutFSS Inst, int Begin = 0>
		AXXEGRO_FORCE_INLINE inline auto At(const Inst& v, int i) -> typename Inst::ElementType const& {
			using VT = Inst&;
			using RetT = typename Inst::ElementType const&;
			return const_cast<RetT>(At(const_cast<VT>(v), i));
		}


	}
}

namespace std {

	template<::iui::vec::InstOfBaseCoords T>
	struct tuple_size<T> : std::integral_constant<std::size_t, T::NumDimensions> {};

	template<std::size_t I, ::iui::vec::InstOfBaseCoords T>
	struct tuple_element<I, T> {
		using type = typename T::ElementType;
	};

	template<std::size_t I, ::iui::vec::InstOfBaseArrCoords T>
	AXXEGRO_FORCE_INLINE inline constexpr auto get(T& v) -> std::tuple_element_t<I, T>& {
		return get<I>(v.coord);
	}

	template<std::size_t I, ::iui::vec::InstOfBaseArrCoords T>
	AXXEGRO_FORCE_INLINE inline constexpr auto get(const T& v) -> std::tuple_element_t<I, T> const& {
		return get<I>(v.coord);
	}

	template<std::size_t I, ::iui::vec::InstOfBaseNamedCoords T>
	AXXEGRO_FORCE_INLINE inline constexpr auto get(T& v) -> std::tuple_element_t<I, T>& {
		static_assert(I>=0 && I<T::NumDimensions, "std::get<al::Vec> subscript out of range");
		return iui::vec::GetNamedCoordAt<T, I>::from(v);
	}

	template<std::size_t I, ::iui::vec::InstOfBaseNamedCoords T>
	AXXEGRO_FORCE_INLINE inline constexpr auto get(const T& v) -> std::tuple_element_t<I, T> const& {
		static_assert(I>=0 && I<T::NumDimensions, "std::get<al::Vec> subscript out of range");
		return iui::vec::GetNamedCoordAt<T, I>::from(v);
	}

}

namespace iui {
	namespace vec {


		inline constexpr bool ShouldImplementWithArr(int numElements) {
			return numElements > 4;
		}

		template<int I, typename T>
		AXXEGRO_FORCE_INLINE inline constexpr auto Get(T& v) -> typename T::ValueType& {
			return std::get<I>(static_cast<typename T::ImplType&>(v));
		}
		template<int I, typename T>
		AXXEGRO_FORCE_INLINE inline constexpr auto Get(const T& v) -> typename T::ValueType const& {
			return std::get<I>(static_cast<const typename T::ImplType&>(v));
		}

		template<typename T, int N>
		using DefaultImpl = std::conditional_t<
			vec::ShouldImplementWithArr(N),
			vec::BaseArrCoords<T,N>,
			vec::BaseNamedCoords<T,N>
		>;

		template<typename Derived>
		struct AddCrossProduct {
			[[nodiscard]] constexpr Derived cross(const Derived& other) const {
				return static_cast<const Derived*>(this)->internalCross(other);
			}
		};

		template<typename T, int N, typename TImpl = DefaultImpl<T, N>>
		struct BaseVec:
			public TImpl,
			public std::conditional_t<N==3, AddCrossProduct<BaseVec<T, N, TImpl>>, Empty>
		{
			template<typename T1>
			friend struct AddCrossProduct;


			static_assert(N > 0);
			static_assert(std::is_same_v<T, typename TImpl::ElementType>, "Invalid implementation type for BaseVec");
			static_assert(N == TImpl::NumDimensions, "Invalid implementation type for BaseVec");


			using ValueType = std::remove_cvref_t<T>;
			using ElementType = std::remove_cvref_t<T>;
			using ImplType = TImpl;
			static constexpr int NumElements = N;
			static constexpr int IsContiguous = (sizeof(BaseVec) == sizeof(T)*NumElements);

			static constexpr bool isIndexValid(int i) {
				return i>0 && i<NumElements;
			}

			constexpr ValueType& operator[](int i) {
				return At(*this, i);
			}

			constexpr const ValueType& operator[](int i) const {
				return At(*this, i);
			}

			template<int I>
			[[nodiscard]] AXXEGRO_FORCE_INLINE constexpr ValueType& getElem() {
				return Get<I>(*this);
			}

			template<int I>
			[[nodiscard]] AXXEGRO_FORCE_INLINE constexpr const ValueType& getElem() const {
				return Get<I>(*this);
			}


			template<std::invocable<ValueType&> Func>
			AXXEGRO_FORCE_INLINE constexpr void forEach(Func fn) {
				for(int i=0; i<NumElements; i++) {
					fn((*this)[i]);
				}
			}

			template<std::invocable<const ValueType&> Func>
			AXXEGRO_FORCE_INLINE constexpr void forEach(Func fn) const {
				for(int i=0; i<NumElements; i++) {
					fn((*this)[i]);
				}
			}

			template<std::invocable<int, ValueType&> Func>
			AXXEGRO_FORCE_INLINE constexpr void forEachEnumerated(Func fn) {
				for(int i=0; i<NumElements; i++) {
					fn(i, (*this)[i]);
				}
			}

			template<std::invocable<int, const ValueType&> Func>
			AXXEGRO_FORCE_INLINE constexpr void forEachEnumerated(Func fn) const {
				for(int i=0; i<NumElements; i++) {
					fn(i, (*this)[i]);
				}
			}

			template<std::invocable<ValueType&, const ValueType&> Func, int Range = NumElements>
			AXXEGRO_FORCE_INLINE constexpr void pairwiseOp(const BaseVec& other, Func fn) {
				static_assert(Range > 0);

				for(int i=0; i<NumElements; i++) {
					fn((*this)[i], other[i]);
				}
			}

			template<std::invocable Op>
			AXXEGRO_FORCE_INLINE [[nodiscard]] constexpr ValueType foldl(auto initial, Op op) const {
				forEach([&](const ValueType& v){op(initial, v);});
				return initial;
			}

			template<std::invocable Op>
			AXXEGRO_FORCE_INLINE [[nodiscard]] constexpr ValueType foldr(auto initial, Op op) const {
				forEach([&](const ValueType& v){op(v, initial);});
				return initial;
			}

			constexpr void fill(ValueType v) {
				forEach([v](ValueType& dv){dv = v;});
			}

			template<std::convertible_to<ValueType>... TValues>
			constexpr void setValues(TValues... values) {
				[&]<std::size_t... Idxs>([[maybe_unused]] std::index_sequence<Idxs...> seq){
					((getElem<Idxs>() = values), ...);
				}(std::make_index_sequence<std::min<int>(sizeof...(TValues), NumElements)>{});
			}


			constexpr BaseVec& operator+=(const BaseVec& rhs) {
				pairwiseOp(rhs, [](ValueType& dst, const ValueType& src){dst += src;});
				return *this;
			}
			constexpr BaseVec& operator-=(const BaseVec& rhs) {
				pairwiseOp(rhs, [](ValueType& dst, const ValueType& src){dst -= src;});
				return *this;
			}
			constexpr BaseVec& operator*=(const ValueType& rhs) {
				forEach([rhs](ValueType& dst){dst *= rhs;});
				return *this;
			}
			constexpr BaseVec& operator/=(const ValueType& rhs) {
				forEach([rhs](ValueType& dst){dst /= rhs;});
				return *this;
			}

			constexpr BaseVec operator+() const {
				return *this;
			}
			constexpr BaseVec operator-() const {
				BaseVec ret = *this;
				ret.forEach([](ValueType& v){v = -v;});
				return ret;
			}

			constexpr BaseVec operator+(const BaseVec& rhs) const {
				BaseVec ret=*this; ret+=rhs; return ret;
			}
			constexpr BaseVec operator-(const BaseVec& rhs) const {
				BaseVec ret=*this; ret-=rhs; return ret;
			}
			constexpr BaseVec operator*(ValueType rhs) const {
				BaseVec ret=*this; ret*=rhs; return ret;
			}
			constexpr BaseVec operator/(ValueType rhs) const {
				BaseVec ret=*this; ret/=rhs; return ret;
			}

			constexpr bool operator==(const BaseVec& rhs) const {
				for(int i=0; i<NumElements; i++) {
					if((*this)[i] != rhs[i]) {
						return false;
					}
				}
				return true;
			}

			constexpr bool operator!=(const BaseVec& rhs) const {
				return !(*this == rhs);
			}

			[[nodiscard]] constexpr bool almostEqual(const BaseVec& rhs, float maxSqDist = 1e-12) const {
				return (rhs - (*this)).sqLength() < maxSqDist;
			}

			[[nodiscard]] constexpr BaseVec hadamard(const BaseVec& other) const {
				BaseVec ret = *this;
				ret.pairwiseOp(other, [](ValueType& dst, const ValueType& src){dst *= src;});
				return ret;
			}

			[[nodiscard]] constexpr double sqLength() const {
				return this->hadamard(*this).foldl(0.0, [](ValueType& acc, const ValueType& val){acc += val;});
			}

			[[nodiscard]] constexpr double length() const {
				return std::sqrt(sqLength());
			}

			[[nodiscard]] constexpr BaseVec normalizedOr(BaseVec val, float tol = 1e-8) const {
				auto len = this->length();
				if(len >= tol) {
					return (*this) / len;
				} else {
					return val;
				}
			}

			[[nodiscard]] constexpr BaseVec normalized(float tol = 1e-8) const {
				return normalizedOr({}, tol);
			}

			[[nodiscard]] constexpr ValueType dot(const BaseVec& rhs) const {
				return (this->hadamard(rhs)).foldl(0.0, [](ValueType& acc, const ValueType& val){acc += val;});
			}

			template<int R>
			[[nodiscard]] constexpr BaseVec rotated() const {
				constexpr int RVal = ((R % N) + N) % N;
				BaseVec ret;

				for(int i=0; i<NumElements; i++) {
					ret[(RVal + i) % N] = (*this)[i];
				}

				return ret;
			}

			template<size_t... LIdxs>
			AXXEGRO_FORCE_INLINE constexpr BaseVec& internalTranspose([[maybe_unused]] std::integer_sequence<std::size_t, LIdxs...> seq) {
				(std::swap(getElem<LIdxs>(), getElem<N - LIdxs - 1>()), ...);
				return *this;
			}

			[[nodiscard]] constexpr BaseVec transposed() const {
				BaseVec ret = *this;
				if constexpr (NumElements > 1) {
					ret.internalTranspose(std::make_index_sequence<NumElements/2>());
				}
				return ret;
			}

			template<typename TValue2, typename TImpl2 = DefaultImpl<TValue2, N>>
			[[nodiscard]] constexpr BaseVec<TValue2, N, TImpl2> as() const {
				BaseVec<TValue2, N> ret;

				for(int i=0; i<NumElements; i++) {
					ret[i] = (*this)[i];
				}
				return ret;
			}


			[[nodiscard]] constexpr BaseVec<float, N> f32() const {
				return as<float>();
			}
			[[nodiscard]] constexpr BaseVec<double, N> f64() const {
				return as<double>();
			}
			[[nodiscard]] constexpr BaseVec<int, N> i() const {
				return as<int>();
			}
			[[nodiscard]] constexpr BaseVec<unsigned, N> u() const {
				return as<unsigned>();
			}




			constexpr BaseVec() {
				fill(0);
			}

			constexpr explicit BaseVec(ValueType v) {
				fill(v);
			}


			/*
			 * ctor that initializes every value (Vec3f v = {1, 2, 3};)
			 * enables constructing from initializer lists without obnoxious casts
			 */
			template<std::convertible_to<ValueType>... TValues>
			requires (sizeof...(TValues) == NumElements && NumElements > 1)
			constexpr BaseVec(TValues... values) {
				setValues(values...);
			}


			/* GLSL-like concatenation (Vec3f v1 = {1, 2}; Vec4f v2 = {v1, 3};) */
			template<typename SrcVecT, std::convertible_to<ValueType> TVal1, std::convertible_to<ValueType>... TValRest>
			requires (
				SrcVecT::NumElements + sizeof...(TValRest) + 1 == NumElements &&
				std::convertible_to<typename SrcVecT::ValueType, ValueType>
			)
			constexpr BaseVec(SrcVecT vec, TVal1 val1, TValRest... rest) {

				constexpr int VN = SrcVecT::NumElements;

				[&]<std::size_t... Idxs>([[maybe_unused]] std::index_sequence<Idxs...> seq){
					((getElem<Idxs>() = vec.template getElem<Idxs>()), ...);
				}(std::make_index_sequence<VN>{});

				getElem<VN>() = val1;

				[&]<std::size_t... Idxs>([[maybe_unused]] std::index_sequence<Idxs...> seq){
					((getElem<Idxs + VN + 1>() = rest), ...);
				}(std::make_index_sequence<NumElements - VN - 1>{});
			}

			constexpr BaseVec(const std::array<ValueType, NumElements>& elements) {
				[this, elements]<std::size_t... Idxs>([[maybe_unused]] std::index_sequence<Idxs...> seq){
					((getElem<Idxs>() = elements[Idxs]), ...);
				}(std::make_index_sequence<NumElements>{});
			}

			/*
			 * between-types conversion, implicit when non-narrowing
			 */
			template<typename OtherVecT>
			requires ((
				!std::same_as<ValueType, typename OtherVecT::ValueType> &&
				NumElements == OtherVecT::NumElements
			))
			explicit (IsNarrowingConversion<ValueType, typename OtherVecT::ValueType>)
			constexpr BaseVec(const OtherVecT& other)
				: BaseVec(other.template as<ValueType, TImpl>())
			{

			}


			constexpr BaseVec(const BaseVec& rhs) = default;
			constexpr BaseVec& operator=(const BaseVec& rhs) = default;
			constexpr BaseVec(BaseVec&& rhs) noexcept = default;
			constexpr BaseVec& operator=(BaseVec&& rhs) noexcept = default;

		private:

			[[nodiscard]] constexpr BaseVec internalCross(const BaseVec& rhs) const {
				static_assert(NumElements == 3);
				auto lr1 = rotated<1>();
				auto lr2 = rotated<2>();
				auto rr1 = rhs.template rotated<1>();
				auto rr2 = rhs.template rotated<2>();
				auto h1 = lr2.hadamard(rr1);
				auto h2 = lr1.hadamard(rr2);
				return h1 - h2;
			}
		};


	}


	template<typename T, int N>
	using Vec = vec::BaseVec<
		T, N,
		std::conditional_t<
			vec::ShouldImplementWithArr(N),
			vec::BaseArrCoords<T,N>,
			vec::BaseNamedCoords<T,N>
		>
	>;

	template<typename T>
	concept VectorType = requires{
		typename T::ValueType;
		typename T::ImplType;
		{T::NumElements} -> std::convertible_to<int>;
	};

	template<typename T>
	using Vec2 = Vec<T, 2>;

	template<typename T>
	using Vec3 = Vec<T, 3>;

	template<typename T>
	using Vec4 = Vec<T, 4>;

	using Vec2d = Vec2<double>;
	using Vec2f = Vec2<float>;
	using Vec2i = Vec2<int>;
	using Vec2u = Vec2<unsigned>;
	using Vec2b = Vec2<uint8_t>;

	using Vec3d = Vec3<double>;
	using Vec3f = Vec3<float>;
	using Vec3i = Vec3<int>;
	using Vec3u = Vec3<unsigned>;
	using Vec3b = Vec3<uint8_t>;

	using Vec4d = Vec4<double>;
	using Vec4f = Vec4<float>;
	using Vec4i = Vec4<int>;
	using Vec4u = Vec4<unsigned>;
	using Vec4b = Vec4<uint8_t>;

	using Vec2i8 = Vec2<int8_t>;
	using Vec2u8 = Vec2<uint8_t>;
	using Vec2i16 = Vec2<int16_t>;
	using Vec2u16 = Vec2<uint16_t>;
	using Vec2i32 = Vec2<int32_t>;
	using Vec2u32 = Vec2<uint32_t>;

	using Vec3i8 = Vec3<int8_t>;
	using Vec3u8 = Vec3<uint8_t>;
	using Vec3i16 = Vec3<int16_t>;
	using Vec3u16 = Vec3<uint16_t>;
	using Vec3i32 = Vec3<int32_t>;
	using Vec3u32 = Vec3<uint32_t>;

	using Vec4i8 = Vec4<int8_t>;
	using Vec4u8 = Vec4<uint8_t>;
	using Vec4i16 = Vec4<int16_t>;
	using Vec4u16 = Vec4<uint16_t>;
	using Vec4i32 = Vec4<int32_t>;
	using Vec4u32 = Vec4<uint32_t>;

}

#endif //AXXEGRO_VEC_HPP