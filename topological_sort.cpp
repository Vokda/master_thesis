#include "topological_sort.hpp"
#include <iostream>
#include <algorithm>
#include "parser.hpp"

//global vars
static Parser parser;
using node = MetaLayer; 

void TopologicalSort::setup(const vector<const LayerParameter*>& lp, const SolverParameter& sp)
{
	cout << "Topological Sort: Creating directed acyclical graph from network description." << endl;
	//creates _graph

	/*
	 * TODO this will not work if a node has multiple neighbors it will get d values from.
	 */
	_delta = 0;

	//creates nodes in directed acyclic graph
	for(auto layer_param : lp)
	{
		//_graph.push_back( graph_node(kv.second.get()) ) );
		_graph.emplace_back(sp, *layer_param);
	}

	//connects node in graph
	//for each node...
	for(itr node = _graph.begin(); node != _graph.end(); node++)
	{
		//against every other node... 
		for(itr node2 = _graph.begin(); node2 != _graph.end(); node2++)
		{
			//for each top...
			for(const auto& top: node->tops)
			{
				//for each bottom...
				for(const auto& bottom: node2->bottoms)
				{
					//if the nodes are not the same...
					if(node != node2)
					{
						//...and the top and bottom string matches (they use the same data)
						//then that means that node must preceed node2 and has it as neighbor.
						if(top == bottom)
						{
							node->set_neighbor(&*node2);
							node2->receive_from.push_back(&top);
						}
					}
					else
						continue;
				}
			}
		}
	}
	/*
	cout << "Topological Sort: The following DAG has been created:" << endl;
	for(itr node = _graph.begin(); node != _graph.end(); node++)
	{
		cout << "Node " << node->get_name() << " has the following neighbors: " << endl;
		for(auto& n : node->neighbors)
		{
			cout << ">" <<  n->get_name() << endl;
		}
	}
	*/
	cout << "Topological Sort: Directed acyclic graph created." << endl;
	//TODO: make sure the graph is DAG
}

const vector<node>& TopologicalSort::operator()()
{
	cout << "Topological Sort: sorting..." << endl;
	dfs();
	cout << "done!" << endl;

	cout << "Setting meta data..." << endl;
	set_meta_data();
	cout << "done!" << endl;

	cout << "Merging layers..." << endl; 
	merge_layers(); //certain layers can be merged. This occurs here.
	cout << "done!" << endl;

	cout << "Setting deltas..." << endl;
	set_delta();
	cout << "done!" << endl;

	cout << endl << "Topological sort: All done!" << endl;
	cout << "Call order result:" << endl;
	for(size_t i = 0; i < _graph.size(); ++i)
	{
		auto& n = _graph[i];
		/* Will print the input type to each leayer
		cout << "<";
		for(auto bottom: n.receive_from)
		{
			//cout << "type: " <<static_cast<std::underlying_type<vector<DataPackage>Type>::type>(bottom->type);
			cout << parser.type_to_string(bottom->type);
		}
		cout << "> ";
		*/
		cout << n.get_name();
		if(n.get_layer_type() == FULLY_CONNECTED or
				n.get_layer_type() == CONVOLUTIONAL)
		{
			cout << '(' << n.get_activation_type() << ')';
		}
		if(i < _graph.size()-1)
			cout << " -> ";
		else
			cout << endl;
		//cout << "discovery time: " << n.discovery_time << endl;
		//cout << "finishing time: " << n.finishing_time << endl;
	}

	for(size_t i = 0; i < _graph.size(); ++i)
	{
		auto& n = _graph[i];

		if(i > 0 and !n.get_send_to().empty())
		{
			cout << n.get_send_to().back()->name;
			cout << " <- ";
		}
		cout << n.get_name();
		if(i < _graph.size()-1 and !n.get_receive_from().empty())
		{
			cout << " <- ";
			cout << n.deltas.back().name << endl;
		}
		else
			cout << endl;
	}
	/*
#ifdef DEBUG
	for(size_t i = 0; i < _graph.size(); ++i)
	{
		cout << "tops" << endl;
		for(auto& t : _graph[i].get_tops())
		{
			cout << t.name << endl;
		}

		cout << "bottom" << endl;
		for(auto& t : _graph[i].get_bottoms())
		{
			cout << t.name << endl;
		}
		cout << "deltas" << endl;
	}
#endif*/
	return _graph;
}

void TopologicalSort::dfs()
{
	//graph_node initiated at construction
	int time = 0;
	for(itr i = _graph.begin(); i != _graph.end(); ++i)
	{
		//cout << "for " << i->get_name() << endl;
		//auto* u = &*i;
		if(i->color == node::Color::WHITE)
		{
			dfs_visit(&*i, time);
		}
	}
}

void TopologicalSort::dfs_visit(node* u, int& time)
{
	//cout << "visiting " << u->get_name() << endl;
	u->color = node::Color::GRAY;
	++time;
	u->discovery_time = time;
	for(auto v : u->neighbors)
	{
		if(v->color == node::Color::WHITE)
		{
			dfs_visit(v, time);
		}
	}
	u->color = node::Color::BLACK;
	u->finishing_time = ++time;
}

void TopologicalSort::set_meta_data()
{ 
	//vector<DataPackage>Type previous_type;
	//assuming that the first layer is the input layer and therefor will determine the type of the input data
	node& input = _graph.front();
	cout << "input layer: " << input.get_name() << endl;
	for(auto& top: input.tops)
	{
		DataSet set = parser.get_data_set(top.name);
		//if data set is unknown check data parameter for source
		if(set == DataSet::UNKNOWN)
		{
			//#ifdef DEBUG
			set = parser.get_data_set(input.layer_parameter.data_param().source());
			cout << "Parser: resolved unknown dataset." << endl;
			cout << "Dataset used: " <<input.layer_parameter.data_param().source()<< endl; 
		}
		top.type = parser.get_data_type(set);
	}

	//for the rest of the layers
	for(size_t i = 1; i < _graph.size(); ++i)
	{
		assert(i < _graph.size());
		//top layer data type
		DataType& top_type = _graph[i].tops.front().type;
		DataType& bottom_type = _graph[i].bottoms.front().type;

#ifdef DEBUG
		cout << "layer: " << _graph[i].get_name() << endl;
		cout << "tops: " << _graph[i].tops.size() << endl;
		cout << "bottoms: "  << _graph[i].bottoms.size() << endl;
#endif

		//TODO so far the other layers will only have one top or less
		//for(auto& top: _graph[i].tops)
		switch(_graph[i].layer_type)
		{
			case LayerType::EUCLIDIEAN: //should not have any tops
			case LayerType::SOFT_MAX:
				break;
			default:
				//top_type = vector<DataPackage>Type::MONOCHROME_IMAGE; 
				top_type = bottom_type;
				break;
		}
	}
}

void TopologicalSort::merge_layers()
{
/*#ifdef DEBUG
	cout << "layers: " << endl;
	for(auto& i: _graph)
	{
		cout << i.get_name() << endl;
	}
#endif*/
	typedef vector<node>::iterator itr;
	std::vector<string> to_remove;
	for(size_t i = 0; i < _graph.size()-1; ++i)
	{
		node& l = _graph[i];
		node& next = _graph[i+1];

		/*
		 * if layer[n+1] is an activation layer it should be merged with the previous layer 
		 * IF it is a convolution or fully connected layer.
		 */
		if(next.is_activation_layer() and 
				(l.get_layer_type() == LayerType::CONVOLUTIONAL or 
				 l.get_layer_type() == LayerType::FULLY_CONNECTED))
		{
			l.merge(next);
			cout << "Merging layer " << next.get_name() << " with "<< l.get_name() << "." << endl;
			to_remove.push_back(next.get_name());
		}
	}
	for(const string& name: to_remove)
	{
		itr node_found = find_if(_graph.begin(), _graph.end(), [&name](node& n){return name == n.get_name();});
		cout << "Removing " << node_found->get_name() << endl;
		_graph.erase(node_found);
	}

#ifdef DEBUG
	cout << "Remaining layers:" << endl;
	for(auto& n: _graph)
	{
		cout << n.get_name() << endl;
	}
#endif
}

void TopologicalSort::set_delta()
{
	for(size_t i = 0; i < _graph.size()-1; ++i)
	{
		node& n = _graph[i];
		node& n2 = _graph[i+1];
		/* 
		 * if node will receive detla values add detla vector
		 * and let the neighor get a pointer to the detla vector
		 * and increment _delta so that all the values get different strings.
		 * node will allocate the data rather than node2 so it will fit inside the
		 * setup()
		 * note: this only works if each layer only will send and receive at most one
		 * delta vector
		 */
		if( (n.get_layer_type() == LayerType::INPUT and n.get_layer_type() == LayerType::EUCLIDIEAN) )
		{}
		else
		{
			string delta_name = "D" + std::to_string(_delta++);
			n.add_delta_data(delta_name);
			n2.send_to.push_back(&n.deltas.back());
#ifdef DEBUG
			cout << "delta_name: " << delta_name << endl;
			cout << n.get_name() << " will receive " << delta_name << " from " << n2.get_name() << endl;
#endif
		}
	}

}
