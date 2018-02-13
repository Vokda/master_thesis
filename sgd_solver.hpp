#ifndef SGD_SOLVER_HPP
#define SGD_SOLVER_HPP
#include "solver.hpp"
#include "skepu_sgd.hpp"

class SGDSolver: public Solver
{
	public:
		SGDSolver(SolverParameter sp, const string& backend);

		//void solve();
		void train();
		//! This will test the network without running the backward function
		void test();

		//!updates weights for fullyconnected layer
		void update_weights(Weights& weights,
                Weights& weights_delta,
                DataPackage& delta,
                DataPackage& bottom, 
				bool bias,
                double bias_value,
                skepu2::Vector<double>& bias_weights, 
				skepu2::Vector<double>& bias_weights_delta);

		//!updates weights for convolutional layers
		void update_weights_conv(DataPackage& weights,
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
                DataPackage* conv_in = nullptr);

		bool training() {return _training;};
		size_t iteration() {return _iteration;};
	private:
		void update_variables(size_t i);
        
        //return true if data should be displayed
        bool display(size_t iteration);
		
		//variables to be remembered through out the run
		double _momentum;
		SkePU_SGD _skepu_sgd;

		size_t _iteration;
		bool _training;
        size_t _max_iter;
        size_t _test_itr;
        bool _display{false};
};
#endif
