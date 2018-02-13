#ifndef EUCLIDIAN_HPP
#define EUCLIDIAN_HPP
#include "layer.hpp"
#include "skepu_euclidean.hpp"
#include "weight_randomizer.hpp"
#include "data_collector.hpp"
#include "solvers.hpp"

//class Solver;


class EuclideanLayer: public Layer
{
	public:
		EuclideanLayer(const MetaLayer& ml, const shared_ptr<skepu2::BackendSpec> spec );

		void setup(vector<DataPackage>& data, map<string, DataPackage*>&, Layer* next);

		void forward();

		void backward(Solver& solver);

		void test_preparation();
		void post_test();

		Weights* get_weights() { return &_weights; }
		const Weights* get_weights() const { return &_weights; }

		void save_state();
		void load_state();

	private:
        
		/**
		 * euclidean parameter has been depricated and this layer will 
		 * have similar function to the fully connected layer which is 
		 * the reason why this parameter exist
		 */
		InnerProductParameter _inner_product_parameter;

		/**
		 * first vector represent the individual neurons
		 * second vector contains the actual weights for a neuron
		 */
		Weights _weights;

		//skepu functionallity 
		SkePU_Euclidean _skepu_f;

		vector<DataPackage>Collector _data_collector;

		//temporary storage for between forward and backward phase
		ProcessedImages _forward_out;
};
#include "euclidean_layer.inl"
#endif
