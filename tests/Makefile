CXX = clang
INCLUDE_PATH = ../
CCFLAGS = -std=c++11 -Wall -Werror -ferror-limit=3 -I$(INCLUDE_PATH) -Wno-mismatched-tags
NVCCFLAGS = -Xcompiler "$(CCFLAGS)"

# this first builds all the tests, then runs them, then prints if everything ran successfully
test : all-test-sources all-test-programs all-test-results
	@echo "All OK."

# this turns a .hpp file into a .cpp file
%.cpp : %.hpp
	@echo '#include <iostream>\n#include "'$<'"\nint main(){ test_'$(basename $<)'(); std::cout << "OK" << std::endl; return 0;}' > $@

# this compiles a .cpp file into an executable file (without a file extension)
% : %.cpp
	$(CXX) $(CCFLAGS) $< -lstdc++ -o $@

# this runs an executable silently
test.% : %
	./$< > /dev/null

# this is a list of test program sources
all-test-sources: $(addsuffix .cpp, $(basename $(wildcard *.hpp)))

# this is a list of test program names
all-test-programs : $(basename $(wildcard *.hpp))

# this is a list of test program names prefixed by test
all-test-results : $(addprefix test., $(basename $(wildcard *.hpp)))

clean:
	rm -f *.cpp $(basename $(wildcard *.hpp))

.PHONY : test clean $(all-test-results)

