#include <cuda.h>
#include <stack>
#include <host_defines.h>

#include "CuME/cume.h"
#include "CuME/cume_variable.h"
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
 *
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

#define VAR_TRUE 1
#define VAR_FALSE -1

#define F_MAX 1
#define F_MIN 0
#define VAR_OFFSET 2


__global__ void
kernel_check_assign(cume::Kernel::Resource *res, int *matrix_formulas, int *assigned, int *var_value, int *sum,
                    int n_var, int n_formulas,
                    int *toAssign, int *hasAssigned) {
    int gtid = res->get_global_tid();
    int i(0);
    while (i < n_formulas && (toAssign[gtid] == 0)) {
        //Si somme de la formule égale à son max et que la variable gtid est dans la formuule, on assigne à la valeur opposé à celle dans la formule
        toAssign[gtid] =
                (assigned[gtid] - 1) * (sum[i] == matrix_formulas[i * (n_var + VAR_OFFSET) + F_MAX]) *
                (matrix_formulas[i * (VAR_OFFSET + n_var) + gtid + VAR_OFFSET]);
        ++i;
    }
    hasAssigned += toAssign[gtid] != 0;
}

//Après ce kernel le tableau toAssign ne contient plus que des zéro
__global__ void
kernel_assign(cume::Kernel::Resource *res, int *assigned, int *var_value, int n_var, int n_formulas,
              int *toAssign, int *hasAssigned) {
    int gtid = res->get_global_tid();
    var_value[gtid] = assigned[gtid] * var_value[gtid] + !assigned[gtid] * toAssign[gtid];
    assigned[gtid] = 1;
    toAssign[gtid] = 0;
}

__global__ void
kernel_compute_sum(cume::Kernel::Resource *res, int *matrix, int *value, int *sum, int n_var, int n_formulas,
                   int *satisfy) { //Compute sum for the gitd formula
    int gtid = res->get_global_tid();
    sum[gtid] = 0;
    int * formule = matrix + gtid  * (VAR_OFFSET + n_var);
    for (int i = 0; i < n_var; ++i) {
        sum[gtid] += formule[i + VAR_OFFSET] == value[i];
    }
    if (sum[gtid] < formule[F_MIN] || sum[gtid] > formule[F_MAX])
        *satisfy = 0;
}

struct node {
    int *assignement_state; // etat de l'assignement avant l'évalution du noeud
    int var;
    int value;
    node *child1;
    node *child2;
    node *parent;
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
    new_node->child1 = NULL;
    new_node->child2 = NULL;
    new_node->value = value;
    new_node->var = var;
    new_node->assignement_state = new int[size];
    std::memcpy(new_node->assignement_state, assignement_state, size);
    return new_node;
}

/**
 * Supprime un noeud, modifie le parent pour supprimer le lien vers ce noeud
 * @param node
 */
void deleteNode(node *node) {
    if (node->parent != NULL) {
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
    //TODO aligner les tableaux sur une puissance de 2 <3
    int var_array_size = n_var;

    cume::Array<int> assigned(var_array_size, 0);
    cume::Array<int> var_value(var_array_size, 0);
    cume::Array<int> toAssign(var_array_size, 0);
    cume::Array<int> sum(var_array_size, 0);
    cume::Variable<int> hasAssign(0);
    cume::Variable<int> satisfy;


    std::stack<node *> toEvaluate;
    std::stack<node *> evaluated;

    node *root1 = createNode(0, VAR_FALSE, NULL, assigned.m_cpu_data, n_var);
    node *root2 = createNode(0, VAR_TRUE, NULL, assigned.m_cpu_data, n_var);

    toEvaluate.push(root1);
    toEvaluate.push(root2);

    bool hasSolution(false);

    cume::Kernel kernelVar(n_var);
    kernelVar.configure(cume::GRID_GUESS | cume::GRID_X, cume::BLOCK_X, 32);
    std::cout << kernelVar << std::endl;

    cume::Kernel kernelFormula(n_formulas);
    kernelFormula.configure(cume::GRID_GUESS | cume::GRID_X, cume::BLOCK_X, 32);
    std::cout << kernelFormula << std::endl;

    while (!toEvaluate.empty() and (not hasSolution)) {
        node *n = toEvaluate.top();
        toEvaluate.pop();
        evaluated.push(n);

        assigned[n->var] = 1;
        var_value[n->var] = n->value;

        do {

            satisfy.value(1);
            satisfy.push();

            //kernel_compute_sum(cume::Kernel::Resource * res, int ** matrix, int * value, int * sum,
            //                   int n_var, int n_formulas, int * satisfy)
            {
                kernel_call(kernel_compute_sum, kernelFormula, &matrix_formulas, &var_value, &sum, n_var, n_formulas,
                            satisfy.m_gpu_value);
            }
            satisfy.pull();
            if (satisfy.value()) {

//                kernel_assign(cume::Kernel::Resource *res, int *assigned, int *var_value, int n_var, int n_formulas,
//                              int *toAssign, int *hasAssigned)
                hasSolution = true;
                break;
            }

            hasAssign.value(0);
            hasAssign.push();

//          kernel_check_assign(cume::Kernel::Resource * res, int **matrix_formulas, int *assigned, int *var_value,
//                              int *sum, int n_var, int n_formulas, int *toAssign, int *hasAssigned) {
            {
                kernel_call(kernel_check_assign, kernelVar, &matrix_formulas, &assigned, &var_value, &sum, n_var,
                            n_formulas, &toAssign, hasAssign.m_gpu_value);
            }
            hasAssign.pull();

            //Si besoin lancer le kernel d'assignement
            if (hasAssign.value()) {
                //TODO kernel assignement
//                kernel_assign(cume::Kernel::Resource *res, int *assigned, int *var_value, int n_var, int n_formulas,
//                              int *toAssign, int *hasAssigned)
                {
                    kernel_call(kernel_assign, kernelVar, &assigned, &var_value, n_var, n_formulas, &toAssign, hasAssign.m_gpu_value);
                }
            }
        } while (hasAssign.value());


        if (!hasSolution) {
            assigned.pull();
            var_value.pull();

            int count_assigned = 0;
            for (auto it = assigned.begin(); it != assigned.end(); ++it) {
                count_assigned += *it;
            }

            if (count_assigned < n_var) {
                //Toutes les variables ne sont pas assignées. Cré
            } else { //toutes les variables sont assignées
                if (!toEvaluate.empty()) {
                    //Il reste des noeuds à évaluer
                    node *nextNode = toEvaluate.top();
                    memcpy(assigned.cpu_addr(), nextNode->assignement_state, var_array_size);
                }
                //Libération de la mémoire
                bool stop(false);
                while ((not stop) and (not evaluated.empty())) {
                    node *toDelete = evaluated.top();
                    if (toDelete->child1 == NULL && toDelete->child2 == NULL) {
                        evaluated.pop();
                        node *parent = toDelete->parent;
                        if (parent->child1 == toDelete) {
                            parent->child1 = NULL;
                        } else {
                            parent->child2 = NULL;
                        }
                        delete toDelete;
                    } else {
                        stop = true;
                    }
                }
            } //Fin if(count_assigned < n_var)
        }//Fin if(!hasSolution)
    }
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

int main(int argc, char * argv[]){
    //TODO tout

    return 0;
}


