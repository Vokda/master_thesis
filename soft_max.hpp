#ifndef SOFT_MAX_HPP
#define SOFT_MAX_HPP
#include "layer.hpp"
#include "skepu_soft_max.hpp"

class Solver;

class SoftMaxLayer: public Layer
{
	public:
		SoftMaxLayer(const MetaLayer& ml, const shared_ptr<skepu2::BackendSpec> spec,
				Data& data, Mapped_data&);

		void setup(Data& data, Mapped_data&, shared_ptr<Layer> next);

		void forward(bool display);

		void backward();

		void update_weights(Solver&) {};

		void test_preparation();
		void post_test();

		void save_state();
		void load_state();

		double get_loss() {return _loss;}
	private:
		shared_ptr<DataPackage> _images;
		shared_ptr<DataPackage> _labels;
		DataPackage _bottom_copy;
		
		//skepu functionallity 
		SkePU_SoftMax _skepu_f;

		//delta, should only be one.
		shared_ptr<DataPackage> _delta;

        //stores loss for one iteration.
		double _loss;

        //used for counting number of displays of data has been made in order to store it correctly
        size_t display_counter{0}; 
};
#endif
