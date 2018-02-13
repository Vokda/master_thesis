#ifndef SKEPU_FUNCTIONS_H
#define SKEPU_FUNCTIONS_H
#include "data_structure.hpp"
#include <memory>
/*
 * TODO
 * when skepu supports taking const arguments, appropriate changes to the function arguments should be made.
 * note:
 * skepu argument order: output arg FIRST then the input args
 * this projects argument order: input args FIRST, output arg LAST
 */

class SkePU
{
	public:
		SkePU(const shared_ptr<skepu2::BackendSpec> spec);

		virtual ~SkePU() {};

		/**
		 * Updates weights according to the formula
		 * w = w + lr*d*a (indices not shown)
		 */
		//TODO for coloured images. Not yet implemented
		//void update_weights(Weights& w, Batch<Delta>& d, Batch<ColouredImage>& a, const double learning_rate);


		/////////////// BASIC MATH FUNCTIONS /////////////////

		/**
		 * Calculates matrix vector multiplication for one vector and one matrix
		 * Needs to be looped over if needed to be run over a whole batch of data
		 */
		void matrix_vector_mul(Weights& w, skepu2::Vector<double>& v, skepu2::Vector<double>& result);

		/** 
         * Summerizes the values in @v
         */
		double sum(skepu2::Vector<double>& v);

		//! sums the abs values in @v
		double sum_abs(skepu2::Vector<double>& v);

		//! Vector vector addition
		void addition(skepu2::Vector<double>& a, skepu2::Vector<double>& b, skepu2::Vector<double>& out);

		//! Vector scalar subtraction
		void subtract_scalar(skepu2::Vector<double>& v, double scalar);

        //! Vector vector subtraction
        void subtraction(skepu2::Vector<double>& a, skepu2::Vector<double>& b, skepu2::Vector<double>& out);

		//! Vector vector multiplication
		//void multiplication(skepu2::Vector<double>& a, skepu2::Vector<double>& b, skepu2::Vector<double>& out);

		//! Dot product of two vectors
		double dot_product(skepu2::Vector<double>& v, skepu2::Vector<double>& w);

		//! Dot product of 1 vector and 1 row of a weight matrix
		double dot_product(skepu2::Vector<double>& v, skepu2::Matrix<double>& w, size_t row);

		//! Calculates the average of value of v
		double average(skepu2::Vector<double>& v);
		double average(skepu2::Matrix<double>& v);
		
		//! Calculate the average error of a delta batch
		vector<double> average_error(skepu2::Vector<double>& d);

		/** Returns the max value
         * Will run on cpu backend for very small sizes of v, mainly for the output layer
         * */
		double find_max(skepu2::Vector<double>& v);

		/**
		 * Copies @N images at @offset in @from to @to. Returns number of images actually copied
		 * If from+offset does not contain enough 
		 */
		void copy(skepu2::Vector<double>::iterator& from_begin,
					skepu2::Vector<double>& destination);

		//arranges the input sequentially accodring to parameters
		void im2col(DataPackage& restructured_output, 
				DataPackage& input,
				size_t kernel_size,
				size_t weights_spatial_size,
				size_t stride,
				size_t nr_neurons,
                size_t i,
				skepu2::BackendSpec* bs = nullptr
				);

        /*!
         * Will pad @input to fit the size of @output.
         * This is specifically made for the backpropagation of convolutional layers.
         */
        void pad_for_output(DataPackage& padded_output,
                DataPackage& input,
                DataPackage& output,
                size_t nr_filters,
                size_t kernel_size,
                size_t stride,
                size_t padding);
                

	protected:

		const shared_ptr<skepu2::BackendSpec> _backend_specification;

		template<typename T>
		bool contains_negative_elements(T& v)
		{
			for(double e: v)
			{
				if(e < 0)
					return true;
			}
			return false;
		}

		//! returns the integer of the 
		int get_int_activation(ActivationType at) const { return at; };

#ifdef DEBUG
		void press_enter_to_continue() 
		{
			cout << "Press enter to continue." << endl;
			cin.get();
		}
#endif

        //variables for padding
        DataPackage _padded_data;
        size_t _side_pad; //difference in (one) dimenson between input and padded output
        size_t _index_start; //where to start copying 
        size_t _index_end; //where to stop copying

	private:
		SkePU() = delete;

};
#endif 
