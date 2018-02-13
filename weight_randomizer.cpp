#include "weight_randomizer.hpp"
#include <skepu2.hpp>
#include <iostream>
#include <chrono>

using namespace std;

/*
[[skepu::userfunction]]
inline double set(double a)
{
	return a;
}

[[skepu::instance]]
static auto set_value = skepu2::Map<1>(set);
*/

WeightRandomizer::WeightRandomizer(int seed)
{
	//set the current time as seed for the number generator
#ifndef DEBUG
	typedef std::chrono::high_resolution_clock hr_clock;
	_generator.seed(hr_clock::now().time_since_epoch().count()); 
#else
	if(seed == -1)
		cout << "Seed not set!" << endl;
	else
	{
		_generator.seed(seed);
		cout << "Seed set to " << seed << endl;
	}
#endif
}

void WeightRandomizer::xavier(skepu2::Matrix<double>& v, double min, double max)
{
	_distribution = uniform_real_distribution<double>(min, max);
	for(double& value: v)
	{
		value = (_distribution(_generator) / (double)v.total_cols());
	}
}

void WeightRandomizer::xavier(skepu2::Vector<double>& v, double min, double max)
{
	_distribution = uniform_real_distribution<double>(min, max);
	for(double& value: v)
	{
		value = (_distribution(_generator) / (double)v.size());
	}
}

double WeightRandomizer::random(double min, double max)
{
	_distribution = uniform_real_distribution<double>(min, max);
	return _distribution(_generator);
}


void WeightRandomizer::gaussian(skepu2::Matrix<double>& w, double mean, double std)
{
	std::normal_distribution<double> dist(mean, std);
	for(double& value: w)
	{
		value = dist(_generator);
	}
}

void WeightRandomizer::gaussian(skepu2::Vector<double>& w, double mean, double std)
{
	std::normal_distribution<double> dist(mean, std);
	for(double& value: w)
	{
		value = dist(_generator);
	}
}

void WeightRandomizer::constant(skepu2::Vector<double>& w, double value)
{
	for(double& v: w)
	{
		v = value;
	}
}

void WeightRandomizer::uniform(skepu2::Matrix<double>& v, double min, double max)
{
	_distribution = uniform_real_distribution<double>(min, max);
	for(double& value: v)
	{
		value = (_distribution(_generator) );
	}
}

void WeightRandomizer::uniform(skepu2::Vector<double>& v, double min, double max)
{
	_distribution = uniform_real_distribution<double>(min, max);
	for(double& value: v)
	{
		value = (_distribution(_generator) );
	}
}
