CXX = clang
NVCC = nvcc
INCLUDES = -I../ -I/usr/local/cuda/include
CCFLAGS = -std=c++14 -Wall -Werror -ferror-limit=3 $(INCLUDES) -Wno-mismatched-tags
NVCCFLAGS = -ccbin=$(CXX) --expt-extended-lambda -rdc=true -arch=sm_35 -Xcompiler "$(CCFLAGS)"

# this gets a list of all directories in this directory
DIRECTORIES = $(sort $(wildcard */))

# this first builds all the tests, then runs them, then prints if everything ran successfully
test : all-test-sources all-cpp-test-programs all-cuda-test-programs all-test-results
	@echo "All OK."

# this turns a .hpp file into a .cpp file
%.cpp : %.hpp
	@echo '#include <iostream>\n#include "'$<'"\nint main(){ test_'$(basename $<)'(); std::cout << "OK" << std::endl; return 0;}' > $@

# this turns a .hpp file into a .cu file
%.cu : %.hpp
	@echo '#include <iostream>\n#include "'$<'"\nint main(){ test_'$(basename $<)'(); std::cout << "OK" << std::endl; return 0;}' > $@

# this compiles a .cpp file into an executable file (without a file extension)
% : %.cpp
	$(CXX) $(CCFLAGS) $< -lstdc++ -L/usr/local/cuda/lib64 -lcudart -o $@

# this compiles a .cpp file into a CUDA executable file (without a file extension)
%-cuda : %.cpp
	$(NVCC) $(NVCCFLAGS) -x cu $< -o $@

# this runs an executable silently
test.% : %
	./$< > /dev/null

# this is a list of test program c++ sources
all-test-sources: $(addsuffix .cpp, $(basename $(wildcard *.hpp)))

# this is a list of test program names
all-cpp-test-programs : $(basename $(wildcard *.hpp))

# this is a list of CUDA test program names
all-cuda-test-programs : $(addsuffix -cuda, $(basename $(wildcard *.hpp)))

# this is a list of test program names prefixed by test
all-test-results : $(addprefix test., $(basename $(wildcard *.hpp))) $(addsuffix -cuda, $(addprefix test., $(basename $(wildcard *.hpp))))

clean:
	rm -f *.cpp *.cu *-cuda $(basename $(wildcard *.hpp))

.PHONY : test clean $(all-test-results) $(DIRECTORIES)

# when the name of a directory is a target, make recursively descends into that directory
$(DIRECTORIES):
	$(MAKE) --directory=$@ $(TARGET)

# the all target tests absolutely everything
all: test $(DIRECTORIES)

# the all_clean target cleans absolutely everything
all_clean: clean
	for dir in $(DIRECTORIES); do \
	  $(MAKE) --directory=$$dir $@; \
	 done

