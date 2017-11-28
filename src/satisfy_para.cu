#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include "gpu_timer.h"
#include "my_cuda.h"


__global__ void satisfy_gpu(int min, int max, bool * satisfy, char * matrice,int CMAX, int LMAX){

	long int ligne = threadIdx.x;

	while(ligne < LMAX){
		unsigned int somme=0;

		for(long int i=0;i<CMAX;i++){

			somme+= matrice[CMAX*ligne+i];

		}

		//On récupère le resultat du test dans un tableau de bool
		satisfy[ligne] = ((somme>=min)&&(somme<=max));

		ligne+=blockDim.x;
	}

}

bool satisfy_tab(bool * satisfy_tab, int size){

	for(int i=0;i<size;i++){

		if(!satisfy_tab[i]){
			return false;
		}

	}

	return true;

}

void transposeMatriceTab(char * m, char * mInverse,int CMAX, int LMAX){

	for(long int j=0;j<CMAX;j++){
		for(long int i=0;i<LMAX;i++){

			mInverse[j * LMAX + i] = m[i*CMAX+j];

		}
	}
}


void remplirMatriceTab(char * matrice, int CMAX, int LMAX){

	for(long int i=0;i<CMAX;i++){

		for(long int j=0;j<LMAX;j++){

			matrice[ i * CMAX + j] = i == j ? 1 : 0;

		}
	}
}

int main(int argc, char ** argv){


	int NB_PIGEONS, NB_PIGEONNIERS;

	//On récupère les arguments
	if((argv[1])&& (argv[2])){
		NB_PIGEONS = atoi(argv[1]);
		NB_PIGEONNIERS = atoi(argv[2]);
	}else
	{
		cout<<"Veuillez remplir en paramètre le nombre de pigeons et de pigeonniers de cette façon :"<<endl;
		cout<<"./gpu_satifsy.exe NB_PIGEONS NB_PIGONNIERS"<<endl;
		exit (EXIT_FAILURE);
	}

	//Les matrices sont des tableaux, plus facile à envoyer vers le device après
	char * cpu_matrice_tab = new char[NB_PIGEONS*NB_PIGEONNIERS];
	char * cpu_matriceTrans = new char[NB_PIGEONS*NB_PIGEONNIERS];


	//Tableau qui permet de récupérer les résultats des tests sur les lignes
	bool * cpu_satisfy_pigeons = new bool[NB_PIGEONS];
	bool * cpu_satisfy_pigeonniers = new bool[NB_PIGEONNIERS];

	char * gpu_matrice_tab;
	char * gpu_matriceTrans;

	bool * gpu_satisfy_pigeons;
	bool * gpu_satisfy_pigeonniers;

	//On alloue
	cuda_check(cudaMalloc((void **)&gpu_matrice_tab,NB_PIGEONS*NB_PIGEONNIERS*sizeof(char)));
	cuda_check(cudaMalloc((void**)&gpu_satisfy_pigeons,NB_PIGEONS*sizeof(bool)));
	cuda_check(cudaMalloc((void**)&gpu_satisfy_pigeonniers,NB_PIGEONNIERS*sizeof(bool)));
	cuda_check(cudaMalloc((void**)&gpu_matriceTrans,NB_PIGEONS*NB_PIGEONNIERS*sizeof(char)));

	//On remplit la matrice du cpu
	remplirMatriceTab(cpu_matrice_tab,NB_PIGEONNIERS,NB_PIGEONS);

	transposeMatriceTab(cpu_matrice_tab,cpu_matriceTrans,NB_PIGEONNIERS,NB_PIGEONS);

	cuda_check(cudaMemcpy(gpu_matrice_tab,cpu_matrice_tab,sizeof(char)*NB_PIGEONS*NB_PIGEONNIERS,cudaMemcpyHostToDevice));
	cuda_check(cudaMemcpy(gpu_matriceTrans,cpu_matriceTrans,sizeof(char)*NB_PIGEONS*NB_PIGEONNIERS,cudaMemcpyHostToDevice));

	GPUTimer g_timer;
	g_timer.start();

	//Teste la contrainte des pigeons sur chaque ligne (un pigeon ne peut être que dans un et un seul pigeonnier)
	satisfy_gpu<<<1,1024>>>(1,1,gpu_satisfy_pigeons,gpu_matrice_tab,NB_PIGEONNIERS,NB_PIGEONS);
	cuda_check_kernel();

	//Teste la contrainte des pigeonniers sur la transposée de la matrice initiale
	//Un pigeonnier peut contenir 0 ou 1 pigeon
	satisfy_gpu<<<1,1024>>>(0,1,gpu_satisfy_pigeonniers,gpu_matriceTrans,NB_PIGEONS,NB_PIGEONNIERS);
	cuda_check_kernel();

	g_timer.stop();


	cuda_check(cudaMemcpy(cpu_satisfy_pigeons,gpu_satisfy_pigeons,sizeof(bool)*NB_PIGEONS,cudaMemcpyDeviceToHost));
	cuda_check(cudaMemcpy(cpu_satisfy_pigeonniers,gpu_satisfy_pigeonniers,NB_PIGEONNIERS*sizeof(bool),cudaMemcpyDeviceToHost));

	cout << satisfy_tab(cpu_satisfy_pigeons, NB_PIGEONS) << endl;
	cout << satisfy_tab(cpu_satisfy_pigeonniers, NB_PIGEONNIERS) << endl;

	cout << "gpu time(ms)=" << g_timer << endl;

	return 0;
}
