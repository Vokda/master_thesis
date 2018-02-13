#ifndef LAYER_FACTORY_H
#define LAYER_FACTORY_H
#include "proto/caffe.pb.h"
#include "data_structure.hpp"
#include "topological_sort.hpp"
#include "parser.hpp"


namespace skepu2
{
	struct BackendSpec;
}
class Layer;
class MetaLayer;

using namespace caffe;
using namespace std;

class LayerFactory
{
	using node = MetaLayer;
	public:

		LayerFactory();

		/*
		 * the function a client will call to obtain a desired layer from the factory based on the node.
		 * If the node represents an input layer then it will be made in this function as it does not take
		 * any template arguments.
		 * No error checking is made in this function. That is done in subsequent functions.
		 */
		Layer* make_layer(const node& ml, const shared_ptr<skepu2::BackendSpec> spec,
				vector<shared_ptr<DataPackage>>& data_ref, Mapped_data& mapped_data) const;

	private:

		/*
		 * Makes layers that require one or more template arguments.
		 * The layer type is determined in this function. 
		 * If layer type cannot be determined a null pointer is returned. 
		 * The tempalte argument is determined by crete_templated_layer()
		 */
		/*
		template<typename T>
		Layer* make_templated_layer(const node& n, const shared_ptr<skepu2::BackendSpec> spec) const;
		*/

		/*
		 * determines the template arguments from the enums provided by node n.
		 * If a template argument can not determined a null pointer is returned.
		 */
		Layer* create_templated_layer(const node& n, const shared_ptr<skepu2::BackendSpec> spec,
				vector<DataType> types, vector<shared_ptr<DataPackage>>&, Mapped_data&) const;

		Parser _parser;

};
//#include "layer_factory.inl"
#endif
