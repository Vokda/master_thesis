#ifndef SKEPU_POOLING_HPP
#define SKEPU_POOLING_HPP
#include "skepu_base.hpp"

namespace caffe
{
	class PoolingParameter;
}

class SkePU_Pooling: public SkePU
{
	public:
		SkePU_Pooling(const shared_ptr<skepu2::BackendSpec>& spec);
	
		//! set the size of the indicies vector (not known at construction of this object)
		void set_nr_indicies(size_t size){ _indicies.resize(size); } 

		void forward(DataPackage& input, DataPackage& output, const caffe::PoolingParameter& pp, 
				size_t nr_neurons);

		void backward(DataPackage& delta_bottom, DataPackage& delta_top, const caffe::PoolingParameter& pp);


	private:
		//stores the index of max values pooled from input
		skepu2::Vector<size_t> _indicies;
};
#endif
