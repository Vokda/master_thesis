#ifndef TOPOLOGICAL_SORT_H
#define TOPOLOGICAL_SORT_H
#include <vector>
#include <memory>
#include "layer.hpp"
#include "meta_layer.hpp"

using namespace std;

class TopologicalSort
{
	private:
		using node = MetaLayer; 
	public:
		//! creates the necessary graph of the DAG to be sorted
		void setup(const vector<const LayerParameter*>& lp, const SolverParameter& sp);
		//void setup(const vector<Layer*>& l);
		//TopologicalSort(const LayerParameter& lp);

		//! sorts the _layers topologically and put the nodes in order in call_order
		const vector<node>& operator()();

		//! get sorted graph.
		const vector<node>& get_graph() const { return _graph; }

		int get_delta() const { return _delta; } 
	private:
		//! performs depth first search
		void dfs();
		void set_meta_data();

		vector<node> _graph;

		typedef typename vector<node>::iterator itr;

		//!visits a node  with depth first.
		void dfs_visit(node*, int& time);

		/**
		 * merges layer that can be merged.
		 * Currently activation layers and the lower layer are merged into one layer.
		 */
		void merge_layers();

		//! assign delta data to layers.
		void set_delta();

		//number of vectors containing delta values necessary
		int _delta;
};
#endif
