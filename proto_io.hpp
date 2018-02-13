//this header reads proto files.

#ifndef PROTO_IO_H
#define PROTO_IO_H

#include <iostream>
#include <fstream>
#include <string>
#include <google/protobuf/text_format.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <stdio.h>
#include <stdexcept>
#include "proto/caffe.pb.h"
#include <list>
#include <tuple>
using namespace std;
using namespace google::protobuf;
using namespace google::protobuf::io;
using namespace caffe;


class Proto_io
{
	private:
		typedef std::tuple<LogLevel, string, int, string> LogMessage ;
		typedef list<LogMessage> LogStack ;
		bool failed;
	public:
		Proto_io()
		{
			GOOGLE_PROTOBUF_VERIFY_VERSION;
			//SetLogHandler(LogHandler);
		}

		~Proto_io()
		{
			ShutdownProtobufLibrary();
		}

		void read_prototxt(const string& filename, Message* msg) //const
		{
			FILE* file = fopen(filename.c_str(), "r");

			/*
			//if file failed to open try adding networks/ to the filename to see if that helps.
			//it could simply be that the user forgot to add the appropropriate directorry
			bool file_found = true;
			if(file == NULL) 
			{
				file_found = false;
				cout << __PRETTY_FUNCTION__ << ": failed to open " << filename;
				cout << ". The function will try to look for the file in the networks/ directory...";
				string new_filename = filename;
				new_filename.insert(0, "networks/");
				file = fopen(new_filename.c_str(), "r");
			}
			*/

			if(file != NULL) //if it does not fail, then great! File found!
			{
				//if(!file_found) //this is for when previous file was not found.
					//cout << "file found!" << endl;
				int file_descriptor = fileno(file);
				FileInputStream fis(file_descriptor);
				fis.SetCloseOnDelete(true);
				if(!TextFormat::Parse(&fis, msg))
				{
					//LogStack e_stack;
					//if(get_error(&e_stack))
					{
						failed =true;
						throw runtime_error("Protobuf could not be parsed.");
					}
				}
				else
					failed =false;

			}
			else
			{
				string base ="basename "+filename;
				//const char* c = command.c_str();
				system(base.c_str());
				throw runtime_error("The file "+base+" not found!");
			}
		}

		Message* read_prototxt(const string& filename) //const
		{
			Message* msg = nullptr;
			FILE* file = fopen(filename.c_str(), "r");

			//if file failed to open try adding networks/ to the filename to see if that helps.
			//it could simply be that the user forgot to add the appropropriate directorry
			bool file_found = true;
			if(file == NULL) 
			{
				file_found = false;
				cout << __PRETTY_FUNCTION__ << ": failed to open " << filename;
				cout << ". The function will try to look for the file in the networks/ directory...";
				string new_filename = filename;
				new_filename.insert(0, "networks/");
				file = fopen(new_filename.c_str(), "r");
			}

			if(file != NULL) //if it does not fail, then great! File found!
			{
				if(!file_found) //this is for when previous file was not found.
					cout << "file found!" << endl;
				int file_descriptor = fileno(file);
				FileInputStream fis(file_descriptor);
				fis.SetCloseOnDelete(true);
				if(!TextFormat::Parse(&fis, msg))
				{
					//LogStack e_stack;
					//if(get_error(&e_stack))
					{
						throw runtime_error("Protobuf could not be parsed.");
					}
				}
			}
			else
			{
				string base ="basename "+filename;
				//const char* c = command.c_str();
				system(base.c_str());
				throw runtime_error("The file "+base+" not found!");
			}
			
			return msg;
		}

		void print(const Message& msg) const
		{
			string s;
			TextFormat::PrintToString(msg, &s);
			cout << s << endl;
		}

	/*private:

		//namespace 
		//{
			LogStack stack;
			bool errno;
		//}

		void* LogHandler(LogLevel lvl, const char* filename, int line, const string& msg)
		{
			stack.push_back({lvl, filename, line, msg});
			errno = true;
		}

		bool get_error(LogStack* s)
		{
			if(errno && s)
			{
				s->assign(stack.begin(), stack.end());
			}

			stack.clear();
			bool old_errno = errno;
			errno = false;

			return old_errno;
		}
		*/
};
#endif
