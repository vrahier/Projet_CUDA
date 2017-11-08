#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include "gpu_timer.h"

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


//Pour le moment la matrice contient des 0 partout sauf au case où i=j
void remplirMatriceTab(int * matrice, int CMAX, int LMAX){

	for(int i=0;i<CMAX;i++){

		for(int j=0;j<LMAX;j++){

			matrice[ i * CMAX + j] = i == j ? 1 : 0;

		}
	}
}

int main(int argc, char ** argv){

	//On récupère les arguments
	const int pigeons = atoi(argv[1]);
	const int pigeonniers = atoi(argv[2]);

	std::cout<<pigeons<<" "<<pigeonniers<<std::endl;

	//Les matrices sont des tableaux, plus facile à envoyer vers le device après
	int * cpu_matrice_tab = new int[pigeons*pigeonniers];
	int * cpu_matriceTrans = new int[pigeons*pigeonniers];


	//Tableau qui permet de récupérer les résultats des tests sur les lignes
	bool * cpu_satisfy_pigeons = new bool[pigeons];
	bool * cpu_satisfy_pigeonniers = new bool[pigeonniers];

	int * gpu_matrice_tab;
	int * gpu_matriceTrans;
	bool * gpu_satisfy_pigeons;
	bool * gpu_satisfy_pigeonniers;

	//On alloue
	cudaMalloc((void **)&gpu_matrice_tab,pigeons*pigeonniers*sizeof(int));
	cudaMalloc((void**)&gpu_satisfy_pigeons,pigeons*sizeof(bool));
	cudaMalloc((void**)&gpu_satisfy_pigeonniers,pigeonniers*sizeof(bool));
	cudaMalloc((void**)&gpu_matriceTrans,pigeons*pigeonniers*sizeof(int));

	//On remplit la matrice du cpu
	remplirMatriceTab(cpu_matrice_tab,pigeonniers,pigeons);


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
	transposeMatriceTab(cpu_matrice_tab,cpu_matriceTrans,pigeonniers,pigeons);


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

	cudaMemcpy(gpu_matrice_tab,cpu_matrice_tab,sizeof(int)*pigeons*pigeonniers,cudaMemcpyHostToDevice);


	//Teste la contrainte des pigeons sur chaque ligne (un pigeon ne peut être que dans un et un seul pigeonnier)
	satisfy_gpu<<<1,pigeons>>>(1,1,gpu_satisfy_pigeons,gpu_matrice_tab,pigeonniers,pigeons);

	cudaMemcpy(cpu_satisfy_pigeons,gpu_satisfy_pigeons,sizeof(bool)*pigeons,cudaMemcpyDeviceToHost);

	cudaMemcpy(gpu_matriceTrans,cpu_matriceTrans,sizeof(int)*pigeons*pigeonniers,cudaMemcpyHostToDevice);

	//Teste la contrainte des pigeonniers sur la transposée de la matrice initiale
	//Un pigeonnier peut contenir 0 ou 1 pigeon
	satisfy_gpu<<<1,pigeonniers>>>(0,1,gpu_satisfy_pigeonniers,gpu_matriceTrans,pigeons,pigeonniers);

	cudaMemcpy(cpu_satisfy_pigeonniers,gpu_satisfy_pigeonniers,pigeonniers*sizeof(bool),cudaMemcpyDeviceToHost);

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
