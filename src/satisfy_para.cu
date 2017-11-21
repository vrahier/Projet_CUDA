#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include "gpu_timer.h"
#include "my_cuda.h"

int NB_PIGEONS =0;
int NB_PIGEONNIERS =0;
//Teste si les contraintes sont respectées pour la matrice sur le device
//Chaque thread s'occupe d'une ligne
__global__ void satisfy_gpu(int min, int max, bool * satisfy, int * matrice,int CMAX, int LMAX){

	int somme=0;

	int indice = threadIdx.x;
	for(int i=0;i<CMAX;i++){

		somme+= matrice[CMAX*indice+i];

	}

	//On récupère le resultat du test dans un tableau de bool
	satisfy[threadIdx.x] = ((somme>=min)&&(somme<=max));

}

//Calcule la transposée d'une matrice (permet de passer de la matrice ligne pigeons colonne pigeonniers à la matrice
//ligne pigeonniers colonne pigeons)
void transposeMatriceTab(int * m, int * mInverse,int CMAX, int LMAX){

	for(int j=0;j<CMAX;j++){
		for(int i=0;i<LMAX;i++){

			mInverse[j * LMAX + i] = m[i*CMAX+j];

		}
	}
}


//Pour le moment la matrice contient des 0 partout sauf au cas où i=j
void remplirMatriceTab(int * matrice, int CMAX, int LMAX){

	for(int i=0;i<CMAX;i++){

		for(int j=0;j<LMAX;j++){

			matrice[ i * CMAX + j] = i == j ? 1 : 0;

		}
	}
}

int main(int argc, char ** argv){

	//On récupère les arguments
	if((argv[1])&& (argv[2])){
		NB_PIGEONS = atoi(argv[1]);
		NB_PIGEONNIERS = atoi(argv[2]);
	}else
	{
		cout<<"veuillez remplir en paramètre le nombre de pigeon et de pigeonniez de cette façon :"<<endl;
		cout<<"./satifsy.exe NB_PIGEONS NB_PIGONNIERS"<<endl;
		exit (EXIT_FAILURE);
	}

	//Les matrices sont des tableaux, plus facile à envoyer vers le device après
	int * cpu_matrice_tab = new int[NB_PIGEONS*NB_PIGEONNIERS];
	int * cpu_matriceTrans = new int[NB_PIGEONS*NB_PIGEONNIERS];


	//Tableau qui permet de récupérer les résultats des tests sur les lignes
	bool * cpu_satisfy_pigeons = new bool[NB_PIGEONS];
	bool * cpu_satisfy_pigeonniers = new bool[NB_PIGEONNIERS];

	int * gpu_matrice_tab;
	int * gpu_matriceTrans;
	bool * gpu_satisfy_pigeons;
	bool * gpu_satisfy_pigeonniers;

	//On alloue
	cuda_check(cudaMalloc((void **)&gpu_matrice_tab,NB_PIGEONS*NB_PIGEONNIERS*sizeof(int)));
	cuda_check(cudaMalloc((void**)&gpu_satisfy_pigeons,NB_PIGEONS*sizeof(bool)));
	cuda_check(cudaMalloc((void**)&gpu_satisfy_pigeonniers,NB_PIGEONNIERS*sizeof(bool)));
	cuda_check(cudaMalloc((void**)&gpu_matriceTrans,NB_PIGEONS*NB_PIGEONNIERS*sizeof(int)));

	//On remplit la matrice du cpu
	remplirMatriceTab(cpu_matrice_tab,NB_PIGEONNIERS,NB_PIGEONS);


	std::cout << "Matrice" << std::endl;

/*	for(int i=0;i<pigeonniers*pigeons;i++){

		std::cout << cpu_matrice_tab[i];

		if(((i+1)%pigeonniers)==0){
			std::cout << std::endl;
		}
	}

	std::cout << std::endl;
	std::cout << std::endl;*/

	//On transpose la matrice pour faire les calculs pour les pigeonniers
	transposeMatriceTab(cpu_matrice_tab,cpu_matriceTrans,NB_PIGEONNIERS,NB_PIGEONS);


	std::cout << "Matrice transposée" << std::endl;

/*	for(int i=0;i<pigeonniers*pigeons;i++){

		std::cout << cpu_matriceTrans[i];

		if(((i+1)%pigeons)==0){

			std::cout << std::endl;

		}
	}

	std::cout << std::endl;
	std::cout << std::endl;
*/


	GPUTimer g_timer;

	g_timer.start();

	cuda_check(cudaMemcpy(gpu_matrice_tab,cpu_matrice_tab,sizeof(int)*NB_PIGEONS*NB_PIGEONNIERS,cudaMemcpyHostToDevice));


	//Teste la contrainte des pigeons sur chaque ligne (un pigeon ne peut être que dans un et un seul pigeonnier)
	satisfy_gpu<<<1,NB_PIGEONS>>>(1,1,gpu_satisfy_pigeons,gpu_matrice_tab,NB_PIGEONNIERS,NB_PIGEONS);
	cuda_check_kernel();

	cuda_check(cudaMemcpy(cpu_satisfy_pigeons,gpu_satisfy_pigeons,sizeof(bool)*NB_PIGEONS,cudaMemcpyDeviceToHost));

	cuda_check(cudaMemcpy(gpu_matriceTrans,cpu_matriceTrans,sizeof(int)*NB_PIGEONS*NB_PIGEONNIERS,cudaMemcpyHostToDevice));

	//Teste la contrainte des pigeonniers sur la transposée de la matrice initiale
	//Un pigeonnier peut contenir 0 ou 1 pigeon
	satisfy_gpu<<<1,NB_PIGEONNIERS>>>(0,1,gpu_satisfy_pigeonniers,gpu_matriceTrans,NB_PIGEONS,NB_PIGEONNIERS);
	cuda_check_kernel();


	cuda_check(cudaMemcpy(cpu_satisfy_pigeonniers,gpu_satisfy_pigeonniers,NB_PIGEONNIERS*sizeof(bool),cudaMemcpyDeviceToHost));

	g_timer.stop();

	std::cout << "Satisfy pigeons" << std::endl;
/*	for(int i=0;i<pigeons;i++){

		std::cout << cpu_satisfy_pigeons[i] << " ";

	}



	std::cout << std::endl;
	std::cout << std::endl;

*/
	std::cout << "Satisfy pigeonniers" << std::endl;

/*	for(int i=0;i<pigeonniers;i++){

		std::cout << cpu_satisfy_pigeonniers[i] << " ";

	}

	std::cout << std::endl;
	std::cout << std::endl;
*/
	cout << "gpu time(ms)=" << g_timer << endl;

	return 0;
}
