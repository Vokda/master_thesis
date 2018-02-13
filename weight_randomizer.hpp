#ifndef WEIGHT_RANDOMIZER_H
#define WEIGHT_RANDOMIZER_H

#include <random>
using namespace std;
namespace skepu2
{
	template<typename T>
	class Matrix;

	template<typename T>
	class Vector;
}


class WeightRandomizer
{
	public:

		WeightRandomizer(int seed=-1);

		/*
		 * xavier distribution. Like uniform but dependant on number of input nodes
		 * To avoid scaling problem the weight is divided with the size of the vector (input size)
		 */
		void xavier(skepu2::Matrix<double>& w, double min, double max);
		void xavier(skepu2::Vector<double>& w, double min, double max);

		//uniform
		double random(double min, double max);
		void uniform(skepu2::Matrix<double>& w, double min, double max);
		void uniform(skepu2::Vector<double>& w, double min, double max);
		

		//gaussian 
		void gaussian(skepu2::Matrix<double>& w, double mean, double std);
		void gaussian(skepu2::Vector<double>& w, double mean, double std);


		//constant weight dist
		void constant(skepu2::Vector<double>& w, double value);
	private:
		::mt19937 _generator;
		uniform_real_distribution<double> _distribution;


		//WeightRandomizer() = delete;
};
#endif
