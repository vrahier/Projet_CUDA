//
// Created by pviolette on 16/12/17.
//

#include <map>
#include <iterator>
#include <sstream>
#include <iostream>
#include <stack>
#include <set>
#include "Solver.h"
#include "Utilities/file.h"


//#define DEBUG
//#define DEBUG_STEP

Solver::Solver() {}

void Solver::parseFile(const char *path) {
    _var_table.clear();

    File reader(path);

    for (auto it = reader.begin(); it != reader.end(); it++) {
        if (it->length() != 0)
            parseLine(*it);
    }
}

Solver::~Solver() {
    for(auto it = _var_table.begin(); it != _var_table.end(); ++it){
        delete it->second;
    }
    for(auto it = _formulas.begin(); it != _formulas.end(); ++it){
        delete *it;
    }
}


void Solver::parseLine(std::string line) {

    std::cout << "Parsing line \"" << line << "\"" << std::endl;

    std::istringstream iss(line);

    std::vector<std::string> tokens;

    std::copy(std::istream_iterator<std::string>(iss),
              std::istream_iterator<std::string>(),
              std::back_inserter(tokens));

    /*
     * Premier et deuxieme mot : min et max
     */

    formula *ptr_formula = new formula;

    this->_formulas.push_back(ptr_formula);

    ptr_formula->id = static_cast<int>(this->_formulas.size() - 1);
    ptr_formula->current_sum = 0;
    ptr_formula->min = std::atoi(tokens[0].c_str());
    ptr_formula->max = std::atoi(tokens[1].c_str());

    for (int i = 2; i < tokens.size(); ++i) {

        formula_var fvar;

        std::string var_name;
        if (tokens[i][0] == '-') {
            fvar.positive = false;
            var_name = tokens[i].substr(1);
        } else {
            fvar.positive = true;
            var_name = tokens[i];
        }

        var *matching_var;
        if (this->_var_table.count(var_name) == 0) {
            std::cout << "Creating var " << var_name << std::endl;
            matching_var = new var;
            matching_var->name = var_name;
            matching_var->assigned = false;
            matching_var->value = false;
            this->_var_table[var_name] = matching_var;
        } else {
            matching_var = this->_var_table.at(var_name);
        }

        matching_var->f.emplace_back(std::pair<formula *, bool>(ptr_formula, fvar.positive));

        fvar.matching_var = matching_var;


        ptr_formula->vars.push_back(fvar);
    }

    ptr_formula->unassigned_var_left = ptr_formula->vars.size();
}

bool Solver::check_satisfy() {
//Soit lance sur le CPU, soit sur le GPU
    for (auto it = this->_formulas.begin(); it != this->_formulas.end(); ++it) {
        if (((*it)->current_sum < (*it)->min) or ((*it)->current_sum > (*it)->max)) {
            return false;
        }
    }
    return true;
}


struct node {
    Solver::var *first_assigned_var; //Variable à laquelle on assigne une valeur à l'évaluation de ce noeud
    bool var_value; //Valeur assigné à l'évaluation
    std::set<Solver::var *> unasigned_var; //variable non assigné avant l'éxecution de ce noeud
    node *childT; //Enfant avec var_value=true
    node *childF; //Enfant avec var_value=true et first_assigned_var identique à celle de childT;
    node *parent;
    bool done; //Vrai si le noeud à été évaluer
};


bool Solver::solve() {
    long int created(0);
    long int next_created(2);
    //Variable non assigné
    std::set<var *> unasigned_vars;
    for (auto it = this->_var_table.begin(); it != this->_var_table.end(); ++it) {
        unasigned_vars.insert(it->second);
    }

    //Noeuds à évaluer
    std::stack<node *> node_stack;

    //Noeuds évalués, à dépiler pour retrouver les états
    std::stack<node *> evaluated_nodes;


    bool solution_found(false);

    //TODO preprocessing : trouver toutes les formules qui ont pour max 0 et forcer les variables de ces formules à faux

    for (auto it = _formulas.begin(); it != _formulas.end(); ++it) {
        if ((*it)->max == 0) {
            for (auto itv = (*it)->vars.begin(); itv != (*it)->vars.end(); ++itv) {
                if (itv->positive) {
                    assign_value(itv->matching_var, false, unasigned_vars);
                } else {
                    assign_value(itv->matching_var, true, unasigned_vars);
                }
            }
        }
    }

    node *root = new node;
    root->parent = NULL;
    root->first_assigned_var = NULL;
    root->childF = NULL;
    root->childT = NULL;
    root->done = false;
    root->unasigned_var = unasigned_vars;
    node_stack.push(root);


    //Recherche depth first. À chaque noeud on assigne un valeur à une variable, puis on essaye de déduire le maximum d'assignation
    bool destack(false);
    while (!node_stack.empty() && !solution_found) {
        if (created == next_created) {
            std::cout << created << std::endl;
            next_created *= 2;
        }
        node *n = node_stack.top();
        node_stack.pop();
        destack = false;
        if (n->done) {
            std::cout << "Node already done ! " << n << std::endl;
            continue;
        }

        evaluated_nodes.push(n);

        n->done = true;
#ifdef DEBUG
        std::cout << "\033[31mEvaluate node " << n->first_assigned_var->name << " " << n->var_value << "\033[0m" << std::endl;
#ifdef DEBUG_STEP
        std::string dummy;
    //    std::getline(std::cin, dummy);
#endif
#endif

        if (n->first_assigned_var != NULL) {
            assign_value(n->first_assigned_var, n->var_value, unasigned_vars);
        }
        bool check = unasigned_vars.empty();
        if (check) { //Toutes les variables sont assignées. Ce noeud est une feuille
            //On vérifie que la solution est correcte

#ifdef DEBUG
            std::cout << "Check satisfy" << std::endl;
            this->display_full(std::cout);
#endif
            if (check_satisfy()) {
                //On a trouver une solution, youpi
                solution_found = true;
            } else {
#ifdef DEBUG
                std::cout << "Not a valid solution" << std::endl;
#ifdef DEBUG_STEP
                std::string dummy;
                std::getline(std::cin, dummy);
#endif
#endif
                if (node_stack.empty()) {
                    //Pas d'autre noeud à evaluer : on en déduit qu'il n'y a pas de solution
                    solution_found = false;
                } else {
                    //Retour en arrière jusqu'a parent du prochain noeud à évaluer dans la pile
                    node *nextNode = node_stack.top();
                    for (auto it = nextNode->unasigned_var.begin(); it != nextNode->unasigned_var.end(); ++it) {
                        if ((*it)->assigned) {
                            remove_assign(*it, unasigned_vars);
                        }
                    }

                    //Libération de la mémoire
                    bool stop(false);
                    while ( (not stop) and (not evaluated_nodes.empty())) {
                            node *toDelete = evaluated_nodes.top();
                            if (toDelete->childT == NULL && toDelete->childF == NULL) {
                                evaluated_nodes.pop();
                                node *parent = toDelete->parent;
                                if (parent->childT == toDelete) {
                                    parent->childT = NULL;
                                } else {
                                    parent->childF = NULL;
                                }
                                delete toDelete;
                            } else {
                                stop = true;
                            }
                    }
                }
            }
        } else {
            //A tester : vérifier quand même la solution avec toutes les variables non assignées à false (on sait jamais, peut être plus rapide)

            //Il reste des variables non assignées
            //On créer donc des enfants
#ifdef DEBUG
            std::cout << "Creating node for " <<  (*(unasigned_vars.begin()))->name << std::endl;
#endif
            n->childT = new node;
            n->childT->first_assigned_var = *(unasigned_vars.begin());
            n->childT->done = false;
            n->childT->var_value = false;
            n->childT->childT = NULL;
            n->childT->childF = NULL;
            n->childT->parent = n;
            n->childT->unasigned_var = unasigned_vars;
            node_stack.push(n->childT);
            n->childF = new node;
            n->childF->first_assigned_var = *(unasigned_vars.begin());
            n->childF->done = false;
            n->childF->var_value = true;
            n->childF->childT = NULL;
            n->childF->childF = NULL;
            n->childF->parent = n;
            n->childF->unasigned_var = unasigned_vars;
            node_stack.push(n->childF);
            created += 2;
//                std::cout << "Created node" << std::endl;
        }
    }

    //Suppression des noeuds
    while(not evaluated_nodes.empty()){
        delete evaluated_nodes.top();
        evaluated_nodes.pop();
    }

    while(not node_stack.empty()){
        delete node_stack.top();
        node_stack.pop();
    }

    return solution_found;
}

void Solver::assign_value(Solver::var *v, bool value, std::set<var *> &unasigned_vars) {
    v->value = value;
    v->assigned = true;

    unasigned_vars.erase(v);
#ifdef DEBUG
    std::cout << "[ASSIGN " << (value ? "TRUE" : "FALSE" ) << " " << v->name << "]" << std::endl;
#endif

    for (auto it = v->f.begin(); it != v->f.end(); ++it) {
        if (it->second ==
            value) { //On assigne true et la variable est positive dans la formule / on assigne false et la variable est négative
            it->first->current_sum++;
        }
        it->first->unassigned_var_left--;
    } //Mise à jour des formules contenant v

    //Mécanisme de déduction pour toute les formules contenant v
    for (auto it = v->f.begin(); it != v->f.end(); ++it) {
        if (it->first->unassigned_var_left > 0) { //Deduction uniquement s'il reste des variable à assigner
            if (it->first->current_sum == it->first->max) {
                //Si on atteint la valeur max pour la formule et qu'il reste des variable à assigné dans cette formule
#ifdef DEBUG
                std::cout << "Maxed formula " << *(it->first) << std::endl;
#endif
                //Alors pour toutes les variables de cette formule
                for (auto itv = it->first->vars.begin(); itv != it->first->vars.end(); ++itv) {
#ifdef DEBUG
                    std::cout << "\t" << itv->matching_var->name << " to " << (!itv->positive ? "true" : "false");
#endif
                    if (!(*itv).matching_var->assigned) { //Qui n'ont pas déjà une valeur assignées
                        assign_value(itv->matching_var, !itv->positive, unasigned_vars);
                        //On assigne true si variable négative (donc not v <-> false dans la formule), false sinon
                    }
                }
            } else if (it->first->current_sum + it->first->unassigned_var_left == it->first->min) {
                //Si la somme de la somme courante et du nombre de variable non assigné restant dans la variable est égale à la somme minimal necessaire
                //Alors pour toute les variables restante dans cette formule
                //On assigne les variables restantes
                for (auto itv = it->first->vars.begin(); itv != it->first->vars.end(); ++itv) {
                    if (!(*itv).matching_var->assigned) {
                        assign_value(itv->matching_var, itv->positive, unasigned_vars);
                    }
                }

            }
        }
    }
}

void Solver::remove_assign(Solver::var *v, std::set<var *> &unasigned_var) {

#ifdef DEBUG
    std::cout << "\033[32mRemove assign on " << v->name << "\033[0m" << std::endl;
#endif
    for (auto it = v->f.begin(); it != v->f.end(); ++it) {
        if (it->second == v->value) {
            it->first->current_sum--;
        }
        it->first->unassigned_var_left++;
    }

    v->value = false;
    v->assigned = false;
    unasigned_var.insert(v);
}

std::ostream &Solver::display_full(std::ostream &stream) {
//Montre toutes les infos
    stream << "Variables : " << std::endl;
    for (auto it = this->_var_table.begin(); it != this->_var_table.end(); ++it) {
        stream << *(it->second) << std::endl;
    }

    stream << "Formulas : " << std::endl;
    for (auto it = this->_formulas.begin(); it != this->_formulas.end(); ++it) {
        stream << **it << std::endl;
    }

    return stream;
}

std::ostream &Solver::display_readable_formula(std::ostream &os) {
    //Montre les formules sous forme lisible
    for (auto it = this->_formulas.begin(); it != this->_formulas.end(); ++it) {
        os << "F_" << (*it)->id << "\t" << (*it)->min << " " << (*it)->max << " ";
        for (auto itv = (*it)->vars.begin(); itv != (*it)->vars.end(); ++itv) {
            if (!itv->positive)
                os << "-";
            os << itv->matching_var->name << " ";
        }
        os << std::endl;
    }
}

std::ostream &operator<<(std::ostream &os, const Solver::var &var) {
    os << "{ 'name': '" << var.name << "', 'assigned' : " << (var.assigned ? "'true' " : "'false'") << ", 'value' : "
       << (var.value ? "'true' " : "'false'") << ", 'f' : [ ";
    for (auto it = var.f.begin(); it != var.f.end(); ++it) {
        os << (*it).first->id << (it + 1 != var.f.end() ? " , " : "");
    }

    os << " ] }";
    return os;
}

std::ostream &operator<<(std::ostream &os, const Solver::formula &formula) {
    os << "{'name': " << formula.id << " 'min': " << formula.min << ", 'max' : " << formula.max << ", 'current_sum' : "
       << formula.current_sum << ", 'unassigned_var_left' : " << formula.unassigned_var_left << ", 'vars': [ ";
    for (auto it = formula.vars.begin(); it != formula.vars.end(); ++it) {
        os << "'" << (it->positive ? "" : "-") << it->matching_var->name << "'"
           << (it + 1 != formula.vars.end() ? " , " : "");
    }
    os << " ] }";
    return os;
}

std::ostream &Solver::display_var_value(std::ostream &os) {
    for (auto it = this->_var_table.begin(); it != this->_var_table.end(); ++it) {
        os << it->first << " : " << (it->second->value ? "VRAI" : "FAUX") << std::endl;
    }

    return os;
}