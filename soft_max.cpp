#include "soft_max.hpp"
#include "solvers.hpp"
#include <milli.hpp>
#include <cmath>

#ifdef SKEPU_CUDA
#include <cuda_profiler_api.h>
#endif

SoftMaxLayer::SoftMaxLayer(const MetaLayer& ml, const shared_ptr<skepu2::BackendSpec> spec,
		Data& data, Mapped_data& mapped_data):
	Layer(ml, spec), 
	_skepu_f(spec, _solver_parameter),
	_bottom_copy("bottom_copy")
{

	//setup bottom

	//input layer setups data first
	_images = setup_bottom(mapped_data, get_bottoms()[0]);
#ifdef DEBUG
	cout << "This should be images: " << endl;
	cout << *_images << endl;
#endif

	//then labels
	_labels = setup_bottom(mapped_data, get_bottoms()[1]);
#ifdef DEBUG
	cout << "This should be labels: " << endl;
	cout << *_labels << endl;
#endif
	/*
	for(const string& name: get_bottoms())
	{
		setup_bottom(mapped_data, name);
	}
	*/

	if(_images->get_total_size() != _labels->get_total_size())
		throw runtime_error("Images and labels are of different sizes!");

	cout << "label_size " << get_bottom_vector().back()->get_image_size() << endl;
}

//////SETUP//////
void SoftMaxLayer::setup(Data& data, Mapped_data& m, shared_ptr<Layer> prev)
{
	if(get_deltas().size() > 1)
		throw runtime_error("Soft max: Too many delta vectors!");
	cout << "setup delta sender" << endl;
	_delta = Layer::setup_delta_sender(m, get_deltas().back());

	_previous_layer = prev;
	assert(_previous_layer->get_nr_neurons() > 0);
}

//////////////////// FORWARD //////////////////////////////////
void SoftMaxLayer::forward(bool display)
{
#ifdef DEBUG
    cout << "RUNNING DEBUG OUTPUT FORWARD" << endl;
	double r = _skepu_f.sum(_images->_data);
	int p_nr_n = _previous_layer->get_nr_neurons();
	r /=p_nr_n; //so that a batch larger than 1 is still ok
	
	if(r > p_nr_n)
	{
		//cout << "output from previous layer: " << _images->_data << endl;
		stringstream ss;
		ss << "Output from previous layer too large: " << r << '>' << _previous_layer->get_nr_neurons();
		throw runtime_error(ss.str());
	}

    double temp = _skepu_f.calculate_softmax_loss(*_labels, *_images, *_delta, _previous_layer->get_activation_type(), _testing, true);
    if(temp == _loss && !_testing)
    {
        cout << "current loss "<< temp << endl;
        cout << "previous loss "<< _loss << endl;
        throw runtime_error("Loss is the same as previous iteration!");
    }
	_loss = temp;
#else //not debug begin

/*#ifdef SKEPU_CUDA
        cudaProfilerStart();
#endif*/
    _loss = _skepu_f.calculate_softmax_loss(*_labels, *_images, *_delta, _previous_layer->get_activation_type(), _testing, display);
    //_loss = 1;
/*#ifdef SKEPU_CUDA
        cudaProfilerStop();
#endif*/
#endif //not debug end


	//assert(round(_skepu_f.sum(_images->_data)) <= _images->get_batch_size());
#ifdef DEBUG
	if(!std::isfinite(_loss))
	{
		cout << "loss " << _loss << endl;
		throw runtime_error("Loss is not finite!");
	}
    if(!_testing)
        assert(_loss >= 0);
#endif
}

//////////////////// BACKWARD //////////////////////////////////
void SoftMaxLayer::backward()
{
	//_loss = _skepu_f.calculate_cross_entropy(*_labels, *_images, *_delta, _previous_layer->get_activation_type());

}

void SoftMaxLayer::save_state()
{
}

void SoftMaxLayer::load_state() 
{
}

void SoftMaxLayer::test_preparation() 
{
	Layer::test_preparation();
    _skepu_f.print_and_clear_loss();
	save_state();
}

void SoftMaxLayer::post_test()
{
    _skepu_f.print_correctness();
	save_state();
}
