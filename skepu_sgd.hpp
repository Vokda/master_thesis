#ifndef SKEPU_SGD_HPP
#define SKEPU_SGD_HPP
#include "skepu_base.hpp"
//class DataCollector;

class SkePU_SGD: public SkePU
{
    public:
        SkePU_SGD(const shared_ptr<skepu2::BackendSpec>& spec);

        //!updates weights for fullyconnected layer
        void update_weights(Weights& w,
                DataPackage& d,
                Weights& w_delta,
                DataPackage& input,
                const double lr, 
                const double m,
                bool bias,
                double bias_value, 
                skepu2::Vector<double>& bias_weights,
                skepu2::Vector<double>& bias_weights_delta);

		/*
		   void update_weights(DataPackage& w, DataPackage& d, DataPackage& a, const double learning_rate,
		   const double momentum);
		   */

		//!updates weights for convolutional layers
		void update_weights_conv(DataPackage& w,
                DataPackage& delta,
                DataPackage& bottom,
                DataPackage& w_delta,
                DataPackage& w_sum,
				double learning_rate,
				double momentum,
                size_t nr_neurons,
                size_t stride,
				bool bias,
                double bias_value,
                skepu2::Vector<double>& bias_weights,
				skepu2::Vector<double>& bias_weights_delta,
                DataPackage* conv_in);

	private:
};
#endif
