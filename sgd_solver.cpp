#include "sgd_solver.hpp"
#include <iostream>
#include <iomanip>
#include "network.hpp"
#include "soft_max.hpp"
#include <milli.hpp>
using namespace std;

SGDSolver::SGDSolver(SolverParameter sp, const string& backend):
	Solver(sp, backend),  _momentum(_solver_parameter.momentum()),
	_skepu_sgd(_backend_specification)
{
	if(!_solver_parameter.has_display())
		throw runtime_error("Display parameter not set!");
	else
		cout << "Display data every " << _solver_parameter.display() << " iteration." << endl;
	_max_iter = (size_t)_solver_parameter.max_iter();
	_test_itr = (size_t)_solver_parameter.test_iter(0);
}

void SGDSolver::train()
{
	cout << "SGD training of network." << endl;
	cout << "Will train for " << _max_iter << " iterations." << endl;
	_training = true;
    size_t epoch = 0;
	for(_iteration = 0; _iteration < _max_iter; ++_iteration)
	{
/*#ifdef DEBUG
        cout << " [iteration "<< iteration << "] ---------" << endl;
#endif*/
        _display = display(_iteration);
		if(_display)
		{
			cout << "--- Training iteration " << _iteration << " ---" << endl;
		}

		//forward
		bool completed_forward = _network->forward(_display);

		if(completed_forward)
			cout << "Epoch " << epoch << " completed!" << endl;

        epoch++;
		//backward
		_network->backward();
		//update weights
		learning_rate_update(_iteration);
		_network->update_weights(*this);

		//display
		if(_display)
		{
			cout << "training time so far " << milli::GetSeconds() << " seconds." << endl;
			//assuming last layer is output layer
			auto output_layer = static_cast<SoftMaxLayer*>(_network->get_layers().back().get());
			cout << "loss " << output_layer->get_loss() << endl;
			cout << "learning rate " << _learning_rate << endl;
			cout << "--- Training iteration " << _iteration << " completed ---" << endl;
		}
	}
	cout << "Training completed. Finished training " << _iteration << " iterations." << endl;
}

void SGDSolver::test()
{
	cout << "Testing of network." << endl;
	_network->test_preparation();
	cout << "Will test for " << _test_itr << " iterations." << endl;
	_training = false;
	for(_iteration = 0; _iteration < _test_itr; ++ _iteration)
	{
        _display = display(_iteration);
		if(_display)
		{
			cout << "--- Test iteration " << _iteration << " ---" << endl;
		}
		//if all data is used break loop. No need to loop over test data
		if(_network->forward(_display))
		{
			break;
		}

		if(_display)
		{
			cout << "testing time so far" << milli::GetSeconds() << endl;
			cout << "--- Test iteration " << _iteration << " completed ---" << endl;
		}
			
	}
	cout << "Test completed: All data used." << endl;
	cout << "Finished testing " << _iteration << " test iterations." << endl;
	_network->post_test();
}

void SGDSolver::update_weights(Weights& weights,
        Weights& weights_delta,
        DataPackage& delta_top,
		DataPackage& bottom, 
		bool bias,
        double bias_value,
        skepu2::Vector<double>& bias_weights, 
		skepu2::Vector<double>& bias_weights_delta)
{
	_skepu_sgd.update_weights(weights, delta_top, weights_delta, bottom, _learning_rate, _momentum, bias,
			bias_value, bias_weights, bias_weights_delta);
}

void SGDSolver::update_weights_conv(DataPackage& weights,
        DataPackage& delta,
        DataPackage& bottom,
        DataPackage& weights_delta,
        DataPackage& weights_sum,
        size_t nr_neurons,
        size_t stride,
        bool bias,
        double bias_value,
        skepu2::Vector<double>& bias_weights,
        skepu2::Vector<double>& bias_weights_delta,
        DataPackage* conv_in)
{
    _skepu_sgd.update_weights_conv(weights,
            delta,
            bottom,
            weights_delta, 
            weights_sum,
            _learning_rate,
            _momentum, 
            nr_neurons,
            stride, 
            bias,
            bias_value,
            bias_weights,
            bias_weights_delta,
            conv_in);
}

bool SGDSolver::display(size_t iteration)
{
#if DEBUG
    return true;
#else
    return iteration % _solver_parameter.display() == 0;
#endif
}
