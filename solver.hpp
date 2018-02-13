#ifndef SOLVER_H
#define SOLVER_H
#include "proto_io.hpp"
#include <memory>
#include "data_structure.hpp"
#include <utility>

namespace skepu2
{
	struct BackendSpec;
}
class Network;

class Solver
{
	public:
		Solver(SolverParameter sp, const string& backend);

		virtual ~Solver();

		//!will simply call train and then test
		virtual void solve();

		/*
		 * for all batches
		 * run a forward and backward phase.
		 * When a forward and backward phase (an interation) has been done for all batches (all data)
		 * then one epoch has been completed
		 */
		//!Run training phase of network
		virtual void train() = 0;
		//!Run the test phase of the network
		virtual void test() = 0;

		//update weights of layers
		//!updates weights for fullyconnected layer
		virtual void update_weights(Weights& weights,
                Weights& weights_delta,
                DataPackage& delta,
				DataPackage& bottom, 
				bool bias,
                double bias_value,
                skepu2::Vector<double>& bias_weights, 
				skepu2::Vector<double>& bias_weights_delta) = 0;

		/**
         * updates weights for convolutional layers
         * last parameter @conv_in is for a special case of batch size == 1
         */
		virtual void update_weights_conv(DataPackage& weights,
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
                DataPackage* conv_in = nullptr) = 0;
		
		const SolverParameter& get_solver_parameter() const
		{
			return _solver_parameter;
		}

		virtual bool training() = 0;
		virtual size_t iteration() = 0;

	private:
		enum class Solver_mode
		{
			CPU_SEQ,
			CPU_OPENMP,
			GPU_OPENCL,
			GPU_CUDA
		};

		enum class LR
		{
			STEP,
			EXP,
			INV
		};

		void setup_lr_policy();
	protected:
		//lr updater
		void learning_rate_update(size_t iteration);
		//vars necessary for the above function
		LR _lr_policy;
		double _learning_rate; 
		double _power;
		size_t _step_size;
		double _gamma;
		

		//contains info for how to solve the problem at hand.
		SolverParameter _solver_parameter; 

		//contains info for structure of neural network
		//NetParameter _network_parameter; 
		std::shared_ptr<skepu2::BackendSpec> _backend_specification;
		Proto_io _io;
		Solver_mode caffe_to_skepu_backend(SolverParameter_SolverMode sm) const;
		std::unique_ptr<Network> _network;	
};

#endif
