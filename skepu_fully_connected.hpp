#ifndef SKEPU_FC_H
#define SKEPU_FC_H
#include "skepu_base.hpp"

class SkePU_FullyConnected: public SkePU
{
	public:
		SkePU_FullyConnected(const shared_ptr<skepu2::BackendSpec>& spec);
		SkePU_FullyConnected() = delete;

		/**
		 * This functions runs the forward function of a fully connected layer
		 */
		void forward(DataPackage& input, skepu2::Matrix<double>& weights, DataPackage& output,
				const ActivationType& at, bool bias, double bias_value, skepu2::Vector<double>& bias_weights,
				double A=1, double B=1);


		/**
		 * Calculates error (delta values)
		 * 
		 */
		void error_calculation(DataPackage& delta_out, DataPackage& delta_in, 
				skepu2::Matrix<double>& weights, ActivationType at, DataPackage& bottom);
	private:
		/*
		TODO: have the skeleton as a member instead of instantiating it every call
		skepu2::impl::MapReduceTEMP<2, double, double, double, double> map_reduce;
		auto map_reduce
		*/
};

#endif
