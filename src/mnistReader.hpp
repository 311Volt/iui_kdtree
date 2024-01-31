
#ifndef MNISTREADER_HPP
#define MNISTREADER_HPP

#include <vector>
#include <iostream>
#include <ranges>
#include <fstream>
#include <string>
#include <algorithm>
#include <format>


struct NumberMNIST {
	std::array<int, 784> image;
	int label;

	void print() const {
		static constexpr char ramp[] = R"^^^^( .-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@)^^^^";
		for(int y=0; y<28; y++) {
			for(int x=0; x<28; x++) {
				int index = std::clamp<int>((image[y*28 + x] * sizeof(ramp)) / 256, 0, sizeof(ramp)-1);
				std::cout << ramp[index];
			}
			std::cout << "\n";
		}
		std::cout << "label: " << label << "\n";
	}
};


inline std::vector<NumberMNIST> readSampleFileMNIST(const std::string& filename) {
	using namespace std::string_literals;

	auto beginsWithADigit = [](const std::string& s) {
		return not s.empty() && std::isdigit(s.at(0));
	};
	auto rangeToStr = [](auto&& r) {
		return std::string{std::ranges::begin(r), std::ranges::end(r)};
	};
	auto toInt = [](const std::string& s) {
		return std::stoi(s);
	};
	auto parseListOfInts = [&](const std::string& ints) {
		auto numbers = ints
			| std::views::split(","s)
			| std::views::transform(rangeToStr)
			| std::views::transform(toInt);
		return std::vector<int>{numbers.begin(), numbers.end()};
	};
	auto intsToMNIST = [](const std::vector<int>& vints) {
		NumberMNIST result;
		auto ints = std::span(vints);
		result.label = ints[0];
		std::ranges::copy(ints.subspan(1), result.image.begin());
		return result;
	};


	std::ifstream inFile(filename);
	if(not inFile.good()) {
		throw std::runtime_error(std::format("no such file: {}", filename));
	}


	auto fileContents = std::string(std::istreambuf_iterator<char>(inFile), std::istreambuf_iterator<char>());
	auto numbers = fileContents
		| std::views::split("\n"s)
		| std::views::transform(rangeToStr)
		| std::views::filter(beginsWithADigit)
		| std::views::transform(parseListOfInts)
		| std::views::transform(intsToMNIST);

	return {numbers.begin(), numbers.end()};
}

#endif //MNISTREADER_HPP
