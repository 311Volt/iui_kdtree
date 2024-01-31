
#ifndef DRYBEANSREADER_HPP
#define DRYBEANSREADER_HPP


#include <vector>
#include <iostream>
#include <ranges>
#include <fstream>
#include <string>
#include <algorithm>
#include <format>


struct DryBean {
	std::array<float, 16> features;
	std::string label;
};


inline std::vector<DryBean> readSampleFileDryBeans(const std::string& filename) {
	using namespace std::string_literals;

	auto beginsWithADigit = [](const std::string& s) {
		return not s.empty() && std::isdigit(s.at(0));
	};
	auto rangeToStr = [](auto&& r) {
		return std::string{std::ranges::begin(r), std::ranges::end(r)};
	};
	auto toFloat = [](const std::string& s) {
		try {
			return std::stof(s);
		} catch (std::invalid_argument& ex) {
			printf("toFloat got %s\n", s.c_str());
			throw ex;
		}
	};
	auto parseListOfNumbers = [&](const std::string& nums) {
		auto numbers = nums
			| std::views::split(","s)
			| std::views::transform(rangeToStr);
		DryBean result;
		auto it = std::ranges::begin(numbers);
		for(int i=0; i<16 && it != std::ranges::end(numbers); it++, i++) {
			result.features[i] = toFloat(*it);
		}
		result.label = *it++;
		return result;
	};



	std::ifstream inFile(filename);
	if(not inFile.good()) {
		throw std::runtime_error(std::format("no such file: {}", filename));
	}


	auto fileContents = std::string(
		std::istreambuf_iterator<char>(inFile),
		std::istreambuf_iterator<char>());
	auto items = fileContents
		| std::views::split("\n"s)
		| std::views::transform(rangeToStr)
		| std::views::filter(beginsWithADigit)
		| std::views::transform(parseListOfNumbers);

	return {items.begin(), items.end()};
}

#endif //DRYBEANSREADER_HPP
