CUFLAGS= --optimize 2 \
	--compiler-options "-O2 -Wall" \
	-gencode arch=compute_30,code=sm_30 \
	-gencode arch=compute_35,code=sm_35 \
	-D_FORCE_INLINES

IEEE_COMPLIANCE=-ftz=false -prec-div=true -prec-sqrt=true

NVCC_FLAGS=$(CUFLAGS) $(IEEE_COMPLIANCE)

all: satisfy gpu_satisfy

satisfy: obj/cpu_timer.o obj/satisfy.o
	g++ -std=c++11 -o bin/satisfy.exe obj/satisfy.o obj/cpu_timer.o
	
obj/satisfy.o: src/satisfy.cpp
	g++ -std=c++11 -o obj/satisfy.o -c src/satisfy.cpp
	
obj/cpu_timer.o: src/cpu_timer.cpp
	g++ -std=c++11 -o obj/cpu_timer.o -c src/cpu_timer.cpp
	
gpu_satisfy: src/satisfy_para.cu obj/gpu_timer.o
	nvcc src/satisfy_para.cu -o bin/gpu_satisfy.exe obj/gpu_timer.o $(NVCC_FLAGS)
	
obj/gpu_timer.o: src/gpu_timer.cu
	nvcc --compile src/gpu_timer.cu -o obj/gpu_timer.o $(NVCC_FLAGS)
	
clean:
	rm -rf bin/*
	rm -rf obj/*