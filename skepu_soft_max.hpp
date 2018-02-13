#ifndef SKEPU_SOFT_MAX_HPP
#define SKEPU_SOFT_MAX_HPP
#include "skepu_base.hpp"
#include "proto/caffe.pb.h"


using namespace caffe;

class SkePU_SoftMax: public SkePU
{
	public:
		SkePU_SoftMax(const shared_ptr<skepu2::BackendSpec>& spec, const SolverParameter& sp);
		/**
		 * Will caclulate the corss entropy (loss) for the input data.
		 * Will assume @soft_max_input si the output of the calculate_softmax()
		 */
		double calculate_cross_entropy(
				DataPackage& target,
				DataPackage& soft_max_input,
				DataPackage& delta,
				bool display);


		/**
		 * Will calculate softmax and cross entropy in one instance
		 * returns loss
		 */
		double calculate_softmax_loss(DataPackage& target, DataPackage& input, DataPackage& delta,
				ActivationType at, bool testing, bool display);

		/** 
		 * calculates the correctness of this batch. 
		 * outputs the correctness of the function
		 */
		void correctness(DataPackage& labels, DataPackage& data, bool testing);


        void print_and_clear_loss();
        void print_correctness();
	private:

        //for batches larger than 1
            //calculates softmax and cross entropy in one merged skeleton
            double softmax_crossentropy(DataPackage& target, DataPackage& input, DataPackage& delta);

            //calculates softmax and correctness in one merged skeleton
            void softmax_correctness(DataPackage& input, DataPackage& labels);

        //for batch size == 1
		/**
		 * Will calculate softmax for the input data.
		 * The input will be modified!
		 */
		void calculate_softmax(DataPackage& input);


		//!unvectorizes label
        void unvectorize_label(DataPackage& label, skepu2::Vector<int>& unvect_label);

        //skepu2::Vector<int> _correct;

        //for cross entropy
        //skepu2::Vector<double> _loss;

        size_t _counter = 0;

        double _loss = -1;
};
#endif
