//Pour éviter les problème d'indexing avec CLion
#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __shared__
#define __constant__
#define __global__
#endif

#include <cuda.h>
#include <stack>
#include <host_defines.h>
#include <map>
#include <ostream>
#include <vector>
#include <cstdlib>


#include "CuME/cume_base.h"
#include "CuME/cume.h"
#include "CuME/cume_variable.h"

#include "gpu_parse_file.h"

/*
 * Matrice des formules m_f, taille n_formulas * (n_var + 2)
 * m_f[i] -> tableau de la formule i
 * m_f[i][0] -> somme minimal pour valider f
 * m_f[i][1] -> somme maximal pour valider f
 * m_f[i][j + 2] -> présence de la variable j -> 1 si j est présente, -1 si (non j) est présente, 0 sinon.
 *
 * Tableau int assigned[n_var] -> assigned[j] = 1 si la variable j à une valeur assigné, 0 sinon
 * Tableau int var_value[n_var] -> var_value[j] = 1 si j est assigné à vrai, -1 si assigné à faux
 *
 * Tableau toAssign[n_var] -> toAssign[i] = 1 si i doit être assigné vrai, -1 si doit être assigné faux, 0 sinon
 *
 * Tableau sum[n_formulas] -> somme actuelle dans la formule f
 *
 * Calcule de la somme avec les variable actuellement assigné dans une formule i
 * Pour j de 0 à n_var
 *     sum += (m_f[i][j+2] == var_value[j]) * assigned
 *
 * Déduire les assignements de variables
 *  Thread pour une variable j
 *  Pour i de 0 à n_formulas - 1
 *      toAssign[j] = ((sum[i] == m_f[i][1]) //On assigne si formule i est maxé
 *                      * (m_f[i][j+2] // Et j présente dans f.
 *                      * (assigned[j] - 1) // égale à -1 si non assigné, 0 si assigné. Permet de renverser le signe de m_f[i][gtid+1]
 *                      //-> Vaut 1  ssi formule maxé et non j (m_f[i][j] == -1) présente dans f et j non assigné
 *                      //-> Vaut -1 sii formule maxé et j (m_f[i][j] == 1) présente dans f et j non assigné
 */

#include "gpu_defines.h"

__global__ void
kernel_check_assign(cume::Kernel::Resource *res, int *matrix_formulas, int *assigned, int *var_value, int *sum,
                    int n_var, int n_formulas,
                    int *toAssign) {
    int gtid = res->get_global_tid();
    int i(0);
    while (i < n_formulas && (toAssign[gtid] == 0)) {
        //Si somme de la formule égale à son max et que la variable gtid est dans la formuule, on assigne à la valeur opposé à celle dans la formule
        toAssign[gtid] =
                (assigned[gtid] - 1) * (sum[i] == matrix_formulas[i * (n_var + VAR_OFFSET) + F_MAX]) *
                (matrix_formulas[i * (VAR_OFFSET + n_var) + gtid + VAR_OFFSET]);

        ++i;
    }
}

//Après ce kernel le tableau toAssign ne contient plus que des zéro
__global__ void
kernel_assign(cume::Kernel::Resource *res, int *assigned, int *var_value, int n_var, int n_formulas,
              int *toAssign) {
    int gtid = res->get_global_tid();
    var_value[gtid] = assigned[gtid] * var_value[gtid] + !assigned[gtid] * toAssign[gtid];
//    printf("assign %d <- %d", gtid, var_value[gtid]);
    assigned[gtid] = (toAssign[gtid] != 0) + assigned[gtid];
    toAssign[gtid] = 0;
}

__global__ void
kernel_compute_sum(cume::Kernel::Resource *res, int *matrix, int *value, int *assigned, int *sum, int n_var,
                   int n_formulas) { //Compute sum for the gitd formula
    int gtid = res->get_global_tid();
    sum[gtid] = 0;
    for (int i = 0; i < n_var; ++i) {
        sum[gtid] += (matrix[gtid * (n_var + VAR_OFFSET) + i + VAR_OFFSET] == value[i]) * assigned[i];
    }
//    printf("Compute sum thread %d -> %d\n", gtid, sum[gtid]);
}

__global__ void kernel_check_sum(cume::Kernel::Resource *res, int *result, int * backtrack, int *sum, int *matrix, int n_formulas,
                                 int n_var){
    *result = 0;
    int i = 0;
    while(i < n_formulas){
//        printf("%d ", sum[i] >= matrix[i * (n_var + VAR_OFFSET) + F_MIN] && sum[i] <= matrix[i * (n_var + VAR_OFFSET) + F_MAX]);
        *result += sum[i] >= matrix[i * (n_var + VAR_OFFSET) + F_MIN] && sum[i] <= matrix[i * (n_var + VAR_OFFSET) + F_MAX];
//        printf("%d\n", *result);
        *backtrack += sum[i] > matrix[i * (n_var + VAR_OFFSET) + F_MAX];
        ++i;
    }
}

__global__ void kernel_check_has_assign(cume::Kernel::Resource * res, int * result, int * toAssign, int n_var){
    *result = 0;
    for(int i = 0; i < n_var; ++i){
        *result += toAssign[i] != 0;
    }
}

struct node {
    node *child1;
    node *child2;
    node *parent;
    int *assignement_state; // etat de l'assignement avant l'évalution du noeud
    int var;
    int value;
};

/**
 * Alloue dynamiquement un nouveau node et l'initialise
 * @param var
 * @param value
 * @param parent noeud parent du noeud à créer
 * @param assignement_state tableau des assignements de variable
 * @param size taille du tableau assignement state
 * @return
 */
node *createNode(int var, int value, node *parent, const int *assignement_state, int size) {
    node *new_node = new node;
//    std::cout << "Created node " << new_node << " with parent " << parent << std::endl;
    new_node->child1 = NULL;
    new_node->child2 = NULL;
    new_node->value = value;
    new_node->parent = parent;
    new_node->var = var;
    new_node->assignement_state = new int[size];
    std::copy(assignement_state, assignement_state + size, new_node->assignement_state);
    return new_node;
}

/**
 * Supprime un noeud, modifie le parent pour supprimer le lien vers ce noeud
 * @param node
 */
void deleteNode(node *node) {
    if (node->parent != NULL) {
//        std::cout << "\t Parent not NULL ->" << node->parent << std::endl;
        if (node->parent->child1 == node) {
            node->parent->child1 = NULL;
        } else {
            node->parent->child2 = NULL;
        }
    }
    delete node->assignement_state;
    delete node;
}

/**
 * @param n_var nombre de variable
 * @param n_formulas nombre de formule
 * @param matrix_formulas matrices des formules, de taille n_formulas * (n_var + 2)
 * @return
 */
bool gpu_solve(int n_var, int n_formulas, cume::Matrix<int> matrix_formulas) {
    //TODO aligner les tableaux sur 128 bits ???
    int var_array_size = n_var;
    int formula_array_size = n_formulas;
    cume::Array<int> assigned(var_array_size, 0);
    cume::Array<int> var_value(var_array_size, 0);
    cume::Array<int> toAssign(var_array_size, 0);
    cume::Array<int> sum(formula_array_size, 0);

    cume::Variable<int> * array_sum_result  = new cume::Variable<int>(0);
    cume::Variable<int> * backtrack = new cume::Variable<int>(0);

//    cume::Devices::get_instance().memory_report(std::cout);
    std::stack < node * > toEvaluate;
    std::stack < node * > evaluated;

    node *root1 = createNode(0, VAR_FALSE, NULL, assigned.cpu_addr(), var_array_size);
    node *root2 = createNode(0, VAR_TRUE, NULL, assigned.cpu_addr(), var_array_size);

    toEvaluate.push(root1);
    toEvaluate.push(root2);

    bool hasSolution(false);

    cume::Kernel kernelVar(n_var);
    kernelVar.configure(cume::GRID_GUESS | cume::GRID_X, cume::BLOCK_X, 32);
    kernelVar.set_timer_needed(false);
    std::cout << kernelVar << std::endl;

    cume::Kernel kernelFormula(n_formulas);
    kernelFormula.set_timer_needed(false);
    kernelFormula.configure(cume::GRID_GUESS | cume::GRID_X, cume::BLOCK_X, 32);
    std::cout << kernelFormula << std::endl;

    cume::Kernel kernelSumArray(1);
    kernelSumArray.set_timer_needed(false);
    kernelSumArray.configure(cume::GRID_GUESS| cume::GRID_X, cume::BLOCK_X, 1);

    sum.push();
    assigned.push();
    var_value.push();
    matrix_formulas.push();


    while (!toEvaluate.empty() and (not hasSolution)) {
        node *n = toEvaluate.top();
        toEvaluate.pop();
        evaluated.push(n);

//        assigned.pull();
//        var_value.pull();

//        std::cout << "\033[31mNode : " << n->var << " - " << n->value << " (" << n << " - parent " << n->parent << ")\033[0m" << std::endl;


        assigned[n->var] = 1;
        var_value[n->var] = n->value;

//        std::cout << "Assigned :\t" << assigned;
//        std::cout << "var_value:\t" << var_value;

        int hasAssign(0);

        cume_push(assigned.m_gpu_data + n->var, assigned.m_cpu_data + n->var, int, 1);
        cume_push(var_value.m_gpu_data + n->var, var_value.m_cpu_data + n->var, int, 1);

        do {

            //Si besoin lancer le kernel d'assignement
            if (hasAssign) {
//                kernel_assign(cume::Kernel::Resource *res, int *assigned, int *var_value, int n_var, int n_formulas,
//                              int *toAssign, int *hasAssigned)
                {
//                    std::cout << "Start kernel assign" << std::endl;
                    kernel_call(kernel_assign, kernelVar, &assigned, &var_value, n_var, n_formulas, &toAssign);
                }
//                assigned.pull();
//                var_value.pull();
//                std::cout << "assigned :\t" << assigned << std::endl;
//                std::cout << "var_value :\t" << var_value << std::endl;
//                std::cout << std::endl;
            }

            //kernel_compute_sum(cume::Kernel::Resource * res, int ** matrix, int * value, int * assigned, int * sum,
            //                   int n_var, int n_formulas, int * satisfy)
            {
                kernel_call(kernel_compute_sum, kernelFormula, &matrix_formulas, &var_value, &assigned, &sum, n_var,
                            n_formulas);
            }

            {
//__global__ void kernel_check_sum(cume::Kernel::Resource *res, int *result, int * backtrack, int *sum, int *matrix, int n_formulas, int n_var)
                kernel_call(kernel_check_sum, kernelSumArray, array_sum_result->m_gpu_value, backtrack->m_gpu_value, &sum, &matrix_formulas, n_formulas, n_var);
            }
            array_sum_result->pull();
//            std::cout << "array_sum_result : " << array_sum_result->m_cpu_value << std::endl;
//            std::cout << "var_value\t" << var_value;
//            std::cout << "assigned\t" << assigned;
//            std::cout << "check " << check << std::endl;
//            std::cout << "sum :  " << sum;
            backtrack->pull();

            if(!backtrack->m_cpu_value) {
//          kernel_check_assign(cume::Kernel::Resource * res, int **matrix_formulas, int *assigned, int *var_value,
//                              int *sum, int n_var, int n_formulas, int *toAssign, int *hasAssigned) {
                {
                    kernel_call(kernel_check_assign, kernelVar, &matrix_formulas, &assigned, &var_value, &sum, n_var,
                                n_formulas, &toAssign);
                }
                toAssign.pull();

                {
//                __global__ void kernel_check_has_assign(cume::Kernel::Resource * res, int * result, int * toAssign, int n_var){

                    kernel_call(kernel_check_has_assign, kernelSumArray, array_sum_result->m_gpu_value, &toAssign,
                                n_var);
                }

                array_sum_result->pull();


                hasAssign = array_sum_result->m_cpu_value;
            }
            //Debug
//            toAssign.pull();
//            std::cout << "hasAssign : " << hasAssign << std::endl;
//            std::cout << "toAssign : " << toAssign;
//            std::cout << "toAssign : " << toAssign;

//            std::cout << "Fin itération" << std::endl;
        } while (hasAssign && !backtrack->m_cpu_value);

        if (!hasSolution) {
            int count_assigned(0);
            if(!backtrack->m_cpu_value) {
                {
//                __global__ void kernel_check_has_assign(cume::Kernel::Resource * res, int * result, int * toAssign, int n_var){

                    kernel_call(kernel_check_has_assign, kernelSumArray, array_sum_result->m_gpu_value, &assigned,
                                n_var);
                }

                array_sum_result->pull();

                count_assigned = array_sum_result->m_cpu_value;
            }
            if (count_assigned < n_var && !backtrack->m_cpu_value) {
                //Toutes les variables ne sont pas assignées. Création des noeuds enfant
                assigned.pull();
                int var(0);
                while (assigned[var] != 0) {
                    var++;
                }
                n->child1 = createNode(var, VAR_FALSE, n, assigned.cpu_addr(), n_var);
                n->child2 = createNode(var, VAR_TRUE, n, assigned.cpu_addr(), n_var);
                toEvaluate.push(n->child1);
                toEvaluate.push(n->child2);

            } else { //toutes les variables sont assignées OU backtrack
                //kernel_compute_sum(cume::Kernel::Resource * res, int ** matrix, int * value, int * assigned, int * sum,
                //                   int n_var, int n_formulas, int * satisfy)
                {
                    kernel_call(kernel_compute_sum, kernelFormula, &matrix_formulas, &var_value, &assigned, &sum, n_var,
                                n_formulas);
                }

                {
//__global__ void kernel_check_sum(cume::Kernel::Resource *res, int *result, int * backtrack, int *sum, int *matrix, int n_formulas, int n_var)
                    kernel_call(kernel_check_sum, kernelSumArray, array_sum_result->m_gpu_value, backtrack->m_gpu_value, &sum, &matrix_formulas, n_formulas, n_var);
                }
                array_sum_result->pull();

                if(array_sum_result->m_cpu_value == n_formulas){
                    hasSolution = true;
                }
                else {
                    if (!toEvaluate.empty()) {
                        //Il reste des noeuds à évaluer
                        node *nextNode = toEvaluate.top();
                        std::copy(&(nextNode->assignement_state[0]), &(nextNode->assignement_state[var_array_size]),
                                  &assigned[0]);
                        assigned.push();
//                    std::cout << "Backtracking" << std::endl;
                    }
                    //Libération de la mémoire
                    bool stop(false);
//                std::cout << "Start free memory" << std::endl;
                    while ((not stop) and (not evaluated.empty())) {
                        node *toDelete = evaluated.top();
//                    std::cout << "Evaluate node " << toDelete << std::endl;
                        if (toDelete->child1 == NULL && toDelete->child2 == NULL) {
                            evaluated.pop();
//                        std::cout << "Delete node " << toDelete << std::endl;
                            deleteNode(toDelete);
//                        std::cout << "Deleted node " << toDelete << std::endl;
                        } else {
                            stop = true;
                        }
                    }
                }
//                std::cout << "End free memory" << std::endl;
            } //Fin if(count_assigned < n_var)
        }//Fin if(!hasSolution)
    }
/*
    var_value.pull();

    for(int i = 0; i < n_var; ++i){
        std::cout << i << " -> " << (var_value[i] == VAR_TRUE ? "TRUE" : "FALSE") << std::endl;
    }
*/
    //Libération de la mémoire
    while (not evaluated.empty()) {
        delete evaluated.top();
        evaluated.pop();
    }
    while (not toEvaluate.empty()) {
        delete toEvaluate.top();
        toEvaluate.pop();
    }

    return hasSolution;
}


int main(int argc, char *argv[]) {
    //TODO tout
    cume::Matrix<int> formules(1, 1);
    int n_var;
    int n_formulas;

    std::string filename = "pb_pigeons_2_2.txt";
    if(argc > 1){
        filename = argv[1];
    }

    parseFile(filename.c_str(), formules, n_var, n_formulas);
/*
    for (int i = 0; i < formules.m_rows; ++i) {
        std::cout << formules.get(i, 0) << " " << formules.get(i, 1) << " | ";
        for (int j = 2; j < formules.m_cols; j++) {
            std::cout << formules.get(i, j) << " ";
        }
        std::cout << std::endl;
    }
*/
//    std::cout << "n_formulas : " << n_formulas << std::endl;
//    std::cout << "n_var : " << n_var << std::endl;

    cume::GPUTimer gpu_timer;
    cume::CPUTimer cpu_timer;

    gpu_timer.start();
    cpu_timer.start();

    bool res = gpu_solve(n_var, n_formulas, formules);

    gpu_timer.stop();
    cpu_timer.stop();

    if (res) {
        std::cout << "\033[32mFound solution to " << filename << "\033[0m" << std::endl;
    } else {
        std::cout << "\033[31mNo solution found" << filename << "\033[0m" << std::endl;
    }
    std::cout << "cpu_timer : " << cpu_timer << std::endl;
    std::cout << "gpu_timer : " << gpu_timer << std::endl;

    return 0;
}