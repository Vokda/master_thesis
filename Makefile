################################################################################
#                     ____  _        ____  _   _   ____                        #
#                    / ___|| | _____|  _ \| | | | |___ \                       #
#                    \___ \| |/ / _ \ |_) | | | |   __) |                      #
#                     ___) |   <  __/  __/| |_| |  / __/                       #
#                    |____/|_|\_\___|_|    \___/  |_____|                      #
#                                                                              #
#                          ~ SkePU 2 main Makefile ~                           #
################################################################################


# ---------------------------------------------------------------------------- #
# Default values which can be changed in Makefile.in

DEBUG_FLAGS = #-DSKEPU_DEBUG=1 -DDEBUG=2
# Default backend set for SkePU precompiler.
BACKENDS = -cuda #-openmp -opencl 
SKEPU_DEFINES = $(DEBUG_FLAGS) -DSKEPU_CUDA #-DSKEPU_OPENMP -DSKEPU_OPENCL 

ifneq (, $(findstring DDEBUG=2, $(DEBUG_FLAGS)))
ifneq (, $(findstring OPENCL, $(SKEPU_DEFINES)))
$(error Cannot precompile for opencl with DEBUG set to 2)
endif
ifneq (, $(findstring CUDA, $(SKEPU_DEFINES)))
$(error Cannot precompile for cuda with DEBUG set to 2)
endif
endif
SKEPU_OPTIONS = 
PRECOMPILER_OPTIONS =

# Backend-specific flags
BACKEND_FLAGS =

# Default OpenCL specific flags (matching a CUDA-provided installation)
OPENCL_FLAGS = -lOpenCL -I /usr/local/cuda/include/

# Default Google Benchmark library paths (empty)
GBENCH_INCL =
GBENCH_LIB =


include Makefile.in

# ---------------------------------------------------------------------------- #
# Compilers.

# Conditionally choose either g++ or nvcc for target compiler,
# depending on whether CUDA is in the list of backends
# This will also be used for non-preprocessed compilation of SkePU programs.
ifneq (,$(findstring cuda, $(BACKENDS)))
CXX = nvcc
else
CXX = g++
endif

# Location of SkePU precompiler binary.
SKEPU = $(LLVM_BIN)/skepu


# ---------------------------------------------------------------------------- #
# Target directories.

SOURCES = $(wildcard *.cpp)
SKEPU_SOURCES = $(filter skepu_%.cpp, $(SOURCES))
OTHER_SOURCES = $(filter-out skepu_%.cpp, $(SOURCES))
PRECOMPILED_SOURCES = $(addprefix $(PRECOMPILED_DIR)/, $(addsuffix _precompiled.$(FILETYPE), $(basename $<)))  #$(addprefix $(PRECOMPILED_DIR)/, $(SKEPU_SOURCES))
PRECOMPILED = $(addprefix $(PRECOMPILED_DIR)/, $(addsuffix _precompiled.$(FILETYPE), $(basename $<)))
#PREPARED_SOURCES += $(PRECOMPILED_DIR)/%.cpp $(PRECOMPILED_DIR)/%.$(FILETYPE)

#OBJECTS = $(addprefix $(OBJECTS_DIR)/, $(notdir $(PRECOMPILED_DIR)/$(wildcard *:.cpp=.o))) 
OBJECTS = $(addprefix $(OBJECTS_DIR)/, $(SOURCES:.cpp=.o))
#PRECOMPILED_OBJECTS = $(addprefix $(OBJECTS_DIR)/, $(SKEPU_SOURCES:.cpp=.o))

HOME_DIR = $(HOME)
PROTO_FILE = proto/caffe.pb.cc
OBJECTS_DIR = objects
SOURCES_DIR = sources
PRECOMPILED_DIR = precompiled
# ---------------------------------------------------------------------------- #
# Compiler flags begin here.

# Flags for precompiler.
SKEPU_FLAGS = $(BACKENDS) $(PRECOMPILER_OPTIONS)
SKEPU_FLAGS += -- -std=c++11 -Wno-expansion-to-defined  
SKEPU_FLAGS += -I $(CLANG_SRC)/lib/Headers
SKEPU_FLAGS += -I $(SKEPU_SRC)/include
SKEPU_FLAGS += $(CLANGTOOL_USER_FLAGS)
SKEPU_FLAGS += $(INCLUDES)


# Activate backend flags for CUDA backend
ifneq (,$(findstring cuda, $(BACKENDS)))
BACKEND_FLAGS += -v -Xcudafe "--diag_suppress=declared_but_not_referenced --diag_suppress=set_but_not_used"
NVCCPASS = -Xcompiler
FILETYPE = cu
else
FILETYPE = cpp
endif

# Activate backend flags for OpenMP backend
ifneq (,$(findstring openmp, $(BACKENDS)))
BACKEND_FLAGS += $(NVCCPASS) -fopenmp
endif

# Activate backend flags for OpenCL backend
ifneq (,$(findstring opencl, $(BACKENDS)))
BACKEND_FLAGS += $(OPENCL_FLAGS)
endif

OPTIMIZATION_LEVEL = $(SKEPU_DEFINES) #-DREMOVE_OUTPUT 
ifneq (, $(findstring DEBUG, $(DEBUG_FLAGS))) #when debugging
OPTIMIZATION_LEVEL = -O0 -g $(SKEPU_DEFINES) #-DREMOVE_OUTPUT 
else #when doing a normal run
OPTIMIZATION_LEVEL = -O3 #-pg #-g
endif

# Flags for target compiler, nvcc requires some specific flags which is the reason for the if statement
ifneq (,$(findstring cuda, $(BACKENDS)))
TARGET_FLAGS = -std=c++11 $(OPTIMIZATION_LEVEL) $(SKEPU_OPTIONS) -I $(SKEPU_SRC)/include -I ./  $(NVCCPASS) $(INCLUDES) $(LDFLAGS) $(BACKEND_FLAGS) -Xcompiler -Wno-attributes
else
TARGET_FLAGS = -std=c++11 $(OPTIMIZATION_LEVEL) $(SKEPU_OPTIONS) -I $(SKEPU_SRC)/include -I ./  $(NVCCPASS) $(INCLUDES) $(LDFLAGS) -Wno-attributes $(BACKEND_FLAGS) --no-pie
endif

TARGET_CPP_FLAGS = -std=c++11 $(OPTIMIZATION_LEVEL) $(SKEPU_OPTIONS) -I $(SKEPU_SRC)/include -I ./  $(INCLUDES) -Wno-attributes $(LDFLAGS) $(BACKEND_FLAGS) #-I /usr/local/cuda/include/
TARGET_CU_FLAGS = -dc -ccbin=g++ -std=c++11 -lineinfo -Xcompiler $(OPTIMIZATION_LEVEL) $(SKEPU_OPTIONS) -I $(SKEPU_SRC)/include -I ./ $(INCLUDES) $(LDFLAGS) -I /usr/local/cuda/include/ $(BACKEND_FLAGS)

BENCH_FLAGS = -I $(GBENCH_INCL) -L $(GBENCH_LIB) -lbenchmark 

# Flags for non-preprocessed compilation.
#SEQ_FLAGS = -std=c++11 -O3 -I $(SKEPU_SRC)/include -Wno-attributes # -fno-rtti -fno-exceptions 


# ---------------------------------------------------------------------------- #
# Make recipies begin here.

# ---------------------------------------------------------------------------- #
# This is project specific

# $@ contains target file name
# $< contains first dependency file name
# $* contains the stem example:
# if target is dir/a.foo.b and target pattern is a.%.b then the stem is dir/foo


INCLUDES  	= -I$(HOME)/include/  -Iproto -I$(HOME)/include/google #-L$(HOME)/lib 
#CFLAGS		+= -c -I../../include -fopenmp  -std=c++11 $(INCLUDES) -g -DSKEPU_OPENMP
#NVCCFLAGS	+= -I../../include -DSKEPU_CUDA -Xcompiler -dlink -x cu $(INCLUDES)
#LDFLAGS		+= -fopenmp -lOpenCL #-lcudart
LDFLAGS += -L$(HOME)/lib/ -lprotobuf
PROTO_FLAGS = $(NVCCPASS) -pthread $(NVCCPASS) -lprotobuf -lpthread
#pkg-config --cflags --libs protobuf # fails if protobuf is not installed 
#pkg-config --libs protobuf

OUT_NAME = skepu_ann
RUN_SCRIPT = run

#clean, make, run
.PHONY: all clearscr fresh clean run line no_precompilation old 

.PRECIOUS: $(PRECOMPILED_DIR)/%.cpp $(PRECOMPILED_DIR)/%.cu


#link everything together
all: $(OBJECTS)
	$(CXX) $(OBJECTS) $(PROTO_FILE) -o $(OUT_NAME) $(TARGET_FLAGS) $(PROTO_FLAGS)
	

#for copied files that can be compiled with g++
$(OBJECTS_DIR)/%.o: $(PRECOMPILED_DIR)/%.$(FILETYPE)
ifneq (,$(findstring cuda, $(BACKENDS)))
	#this is specifically for when a skepu struct's operator= needs to be declared __device__
	#for cuda backend only
	sed -i'' 's/\(.\+value&[ ]operator=\)/__device__ \1/g' $<
	#compilation
	nvcc -c $< -o $@ $(TARGET_CU_FLAGS) -DSKEPU_PRECOMPILED $(SKEPU_DEFINES)
else
	g++ -c $< -o $@ $(TARGET_FLAGS) -DSKEPU_PRECOMPILED $(SKEPU_DEFINES) 
endif
	@echo ===================================


#files not containing skepu code can simply copied to the precompiled directory
$(PRECOMPILED_DIR)/%.$(FILETYPE): %.cpp
	case $< in \
	"skepu_"*) $(SKEPU) -name $(basename $<) $< -dir $(PRECOMPILED_DIR)  $(SKEPU_FLAGS);; \
	*) cp $< $@;; \
	esac

fresh: clean clearscr all #run 

clearscr:
	clear
	@printf "Everything cleaned!\n\n\nCompiling everything..."
	@echo $(\e#8\e[1B\e[J) #is supposed to print a line in the terminal

#run:
#	./$(RUN_SCRIPT)

# Deletes all temporary files (including all precompiled sources) and binaries.
clean:
	rm -f $(OBJECTS_DIR)/*
	rm -f $(OUT_NAME)
	rm -f $(PRECOMPILED_DIR)/*
	rm -rf doxygen

