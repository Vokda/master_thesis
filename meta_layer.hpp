#ifndef META_LAYER_HPP
#define META_LAYER_HPP
#include <vector>
#include <string>
#include "data_structure.hpp"
#include "parser.hpp"
#include "proto/caffe.pb.h"

using namespace caffe;

/**
 * The meta layer  used by topological sort class to sort the layers in topological order. 
 * This struct used to be smaller and grew to a very large structure. 
 * This is why it breaks some coding standards such as memeber variables not beginning with '_'
 */
class MetaLayer
{
	public:
		struct meta_data
		{
			bool operator==(const meta_data& other) const
			{
				return this->name == other.name;
			}

			string name = "NO NAME";
			DataType type = DataType::NOT_SUPPORTED;
		};

		MetaLayer(const SolverParameter& sp, const LayerParameter& lp);

		MetaLayer(const MetaLayer& m);

		MetaLayer& operator=(MetaLayer&& m);

		meta_data* add_delta_data(string s);

		void set_neighbor(MetaLayer* n)
		{
			neighbors.push_back(n);
		}

		void set_activation_type(LayerType t);

		void merge(MetaLayer& mergee)
		{
			this->_activation_type = mergee.get_activation_type();
			this->neighbors = mergee.neighbors;
			this->receive_from = mergee.receive_from;
		}

		// ACCESSORS
		LayerType get_layer_type() const { return layer_type; }

		bool is_activation_layer() const;

		const string& get_name() const
		{
			return layer_parameter.name();
		}

		const ActivationType& get_activation_type() const
		{
			return _activation_type;
		}

		const LayerParameter& get_layer_parameter() const { return layer_parameter; }
		const SolverParameter& get_solver_parameter() const { return solver_parameter; }

		const vector<meta_data>& get_tops() const { return tops; }
		const vector<meta_data>& get_bottoms() const { return bottoms; }
		const vector<meta_data>& get_deltas() const { return deltas; }

		const vector<const meta_data*>& get_receive_from() const  { return receive_from; } 
		const vector<const meta_data*>& get_send_to() const { return send_to; } 

	private:
		enum class Color{WHITE, GRAY, BLACK};
		LayerParameter layer_parameter;
		const SolverParameter& solver_parameter;
		int discovery_time;
		int finishing_time;
		Color color;
		LayerType layer_type;
		ActivationType _activation_type;

		vector<MetaLayer*> neighbors;
		//this vector is used for when setting the meta data type
		vector<const meta_data*> receive_from; //bottom
		//both tops and bottoms are gotten from the proto buffer
		vector<meta_data> tops; 
		vector<meta_data> bottoms; //not to be used outside of the MetaLayer
		//separate vector for delta values
		vector<meta_data> deltas;
		//where to send the delta values.
		vector<const meta_data*> send_to;

		/*
		 * friend because of MetaLayer used to be local struct
		 * to avoid changes to topological sort
		 */
		friend class TopologicalSort; 
};

#endif
