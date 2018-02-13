#ifndef POOLING_LAYER_HPP
#define POOLING_LAYER_HPP
#include "layer.hpp"
#include "skepu_pooling.hpp"
//#include "solvers.hpp"

class Solver;
class MetaLayer;

using namespace std;

class PoolingLayer: public Layer
{
	public:
		PoolingLayer(const MetaLayer& ml, const shared_ptr<skepu2::BackendSpec> spec, 
				Data& data, Mapped_data& );
		//PoolingLayer();

		void setup(Data& data, Mapped_data&, shared_ptr<Layer> prev);

		void forward(bool display);
		void backward();
		void update_weights(Solver& s){/*no weights in pooling layer to update*/};
		void test_preparation();

		void save_state();
		void load_state();

	private:

		PoolingParameter _pooling_parameter;

		//skepu functionallity 
		SkePU_Pooling _skepu_f;

		skepu2::Vector<size_t> _output_dims;

		//tops and bottoms
		shared_ptr<DataPackage> bottom_ptr;
		shared_ptr<DataPackage> top_ptr;
};
#endif
