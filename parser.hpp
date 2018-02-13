#ifndef PARSER_H
#define PARSER_H
#include "data_structure.hpp"
#include <map>

class Parser
{
	public:
		Parser();

		/*
		 * returns data set from the data sets name. 
		 * The name can be found in the protobuffer
		 * Can return label too, this is for the meta data.
		 */
		DataSet get_data_set(string data_name) const;

		//returns data type from the data set
		DataType get_data_type(DataSet data) const;

		//get a printable version of data
		string type_to_string(DataType) const;

		//returns LayerType given a string
		LayerType get_layer_type(string layer_name) const;

		//returns string of activation type
		static string activation_f_to_string(ActivationType A);

	private:
		map<DataType, string> _type_strings;
		map<string, LayerType> _layer_map;
		map<string, DataSet> _string_types;
};
#endif
