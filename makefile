CUFLAGS= --optimize 2 \
	--compiler-options "-O2 -Wall" \
	-gencode arch=compute_30,code=sm_30 \
	-gencode arch=compute_35,code=sm_35 \
	-D_FORCE_INLINES

IEEE_COMPLIANCE=-ftz=false -prec-div=true -prec-sqrt=true

NVCC_FLAGS=$(CUFLAGS) $(IEEE_COMPLIANCE)

all: satisfy gpu_satisfy

satisfy: bin/cpu_timer.o bin/satisfy.o
	g++ -std=c++11 -o satisfy bin/satisfy.o bin/cpu_timer.o
	
bin/satisfy.o: src/satisfy.cpp
	g++ -std=c++11 -o bin/satisfy.o -c src/satisfy.cpp
	
bin/cpu_timer.o: src/cpu_timer.cpp
	g++ -std=c++11 -o bin/cpu_timer.o -c src/cpu_timer.cpp
	
gpu_satisfy: src/satisfy_para.cu bin/gpu_timer.o
	nvcc src/satisfy_para.cu -o gpu_satisfy bin/gpu_timer.o $(NVCC_FLAGS)
	
bin/gpu_timer.o: src/gpu_timer.cu
	nvcc --compile src/gpu_timer.cu -o bin/gpu_timer.o $(NVCC_FLAGS)
	
clean:
	rm -rf bin/*