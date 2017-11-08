CUFLAGS= --optimize 2 \
	--compiler-options "-O2 -Wall" \
	-gencode arch=compute_30,code=sm_30 \
	-gencode arch=compute_35,code=sm_35 \
	-D_FORCE_INLINES

IEEE_COMPLIANCE=-ftz=false -prec-div=true -prec-sqrt=true

NVCC_FLAGS=$(CUFLAGS) $(IEEE_COMPLIANCE)

all: satisfy gpu_satisfy

satisfy: cpu_timer.o satisfy.o
	g++ -std=c++11 -o satisfy satisfy.o cpu_timer.o
	
satisfy.o: src/satisfy.cpp
	g++ -std=c++11 -o satisfy.o -c src/satisfy.cpp
	
cpu_timer.o: src/cpu_timer.cpp
	g++ -std=c++11 -o cpu_timer.o -c src/cpu_timer.cpp
	
gpu_satisfy: src/satisfy_para.cu gpu_timer.o
	nvcc src/satisfy_para.cu -o gpu_satisfy gpu_timer.o $(NVCC_FLAGS)
	
gpu_timer.o: src/gpu_timer.cu
	nvcc --compile src/gpu_timer.cu -o gpu_timer.o $(NVCC_FLAGS)
	
clean:
	rm -rf *.o