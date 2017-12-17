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


#define DEBUG
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

    ptr_formula->id = this->_formulas.size() - 1;
    ptr_formula->actual_sum = 0;
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
            matching_var->id = var_name;
            matching_var->assigned = false;
            matching_var->value = false;
            this->_var_table[var_name] = matching_var;
        } else {
            matching_var = this->_var_table.at(var_name);
        }

        matching_var->f.push_back(std::pair<formula*, bool>(ptr_formula, fvar.positive));

        fvar.matching_var = matching_var;


        ptr_formula->vars.push_back(fvar);
    }
}

bool Solver::check_satisfy() {
//Soit lance sur le CPU, soit sur le GPU
    for(auto it = this->_formulas.begin(); it != this->_formulas.end(); ++it){
        if(((*it)->actual_sum < (*it)->min) or ((*it)->actual_sum > (*it)->max )){
            return false;
        }
    }
    return true;
}


struct node {
    Solver::var * first_assigned_var; //Variable à laquelle on assigne une valeur à l'évaluation de ce noeud
    bool var_value; //Valeur assigné à l'évaluation
    std::stack<Solver::var *> assigned_vars; //Liste des variable assigné par la déduction
    node * childT; //Enfant avec var_value=true
    node * childF; //Enfant avec var_value=true et first_assigned_var identique à celle de childT;
    bool done; //Vrai si le noeud à été évaluer
};

bool Solver::solve() {
    //Putin ça va chier ici

    //Variable non assigné
    std::set<var *> unasigned_vars;
    for(auto it = this->_var_table.begin(); it != this->_var_table.end(); ++it){
        unasigned_vars.insert(it->second);
    }

    //Noeuds à évaluer
    std::stack<node*> node_stack;

    //Noeuds évalués, à dépiler pour retrouver les états
    std::stack<node*> evaluated_nodes;

    bool var_value = false;
    node * n = new node;
    n->first_assigned_var = this->_var_table.begin()->second;
    n->done = false;
    n->var_value = var_value ;
    node_stack.push(n);
    n = new node(*n);
    n->var_value = !var_value;
    node_stack.push(n);

    bool solution_found(false);

    //TODO preprocessing : trouver toutes les formules qui ont pour max 0 et forcer les variables de ces formules à faux

    //Recherche depth first. À chaque noeud on assigne un valeur à une variable, puis on essaye de déduire le maximum d'assignation
    while(!node_stack.empty() && !solution_found){
        n = node_stack.top();
        node_stack.pop();

        evaluated_nodes.push(n);

        n->done = true;
#ifdef DEBUG
        std::cout << "\033[31mEvaluate node " << n->first_assigned_var->id << " " << n->var_value << "\033[0m" << std::endl;
#ifdef DEBUG_STEP
        std::string dummy;
    //    std::getline(std::cin, dummy);
#endif
#endif
        assign_value(n->first_assigned_var, n->var_value, unasigned_vars, n->assigned_vars);

        if(unasigned_vars.empty()){ //Toutes les variables sont assignées. Ce noeud est une feuille
            //On vérifie que la solution est correcte
#ifdef DEBUG
            std::cout << "Check satisfy" << std::endl;
            this->display_full(std::cout);
#endif
            if(check_satisfy()){
                //On a trouver une solution, youpi
                solution_found = true;
            }
            else{
#ifdef DEBUG
                std::cout << "Not a valid solution" << std::endl;
#ifdef DEBUG_STEP
                std::string dummy;
        std::getline(std::cin, dummy);
#endif
#endif
                node * unpacked  = n;
                bool stop(false);
                while( (not stop) and (not evaluated_nodes.empty()) ){
#ifdef DEBUG
                    std::cout << "Unpacking " << unpacked->first_assigned_var->id << " " << unpacked->var_value << std::endl;
#endif
                    while(not unpacked->assigned_vars.empty()){
                        remove_assign(unpacked->assigned_vars.top(), unasigned_vars);
                        unpacked->assigned_vars.pop();
                    }
#ifdef DEBUG
                    std::cout << "End unpacking " <<unpacked->first_assigned_var->id << " " << unpacked->var_value << std::endl;
#endif
                    if(!evaluated_nodes.empty()){
                        std::cout << evaluated_nodes.top() << std::endl;
                        unpacked = evaluated_nodes.top();
                        evaluated_nodes.pop();
                        std::cout << unpacked->first_assigned_var->id << std::endl;
                        if(unpacked->childT != NULL){
                            std::cout << "F"<<  unpacked->childT << std::endl;
                            stop = stop or (not unpacked->childT->done);
                        }
                        if(unpacked->childF != NULL){
                            std::cout << unpacked->childF << std::endl;
                            stop = stop or (not unpacked->childF->done);
                        }
                    }else{
                        stop = true;
                    }
#ifdef DEBUG
                    std::cout << "End iteration"<< std::endl;
#endif
                };
            }
        }else{
            //A tester : vérifier quand même la solution avec toutes les variables non assignées à false (on sait jamais, peut être plus rapide)

            //Il reste des variables non assignées
            //On créer donc des enfants
#ifdef DEBUG
            std::cout << "Creating node for " <<  (*(unasigned_vars.begin()))->id << std::endl;
#endif
            n->childT = new node;
            n->childT->first_assigned_var = *(unasigned_vars.begin());
            n->childT->done = false;
            n->childT->var_value = n->var_value;
            n->childT->childT = NULL;
            n->childT->childF = NULL;
            node_stack.push(n->childT);
            n->childF = new node(*n->childT);
            n->childF->var_value = !n->var_value;
            node_stack.push(n->childF);
        }
    }

    return solution_found;
}

void Solver::assign_value(Solver::var *v, bool value, std::set<var *>& unasigned_vars, std::stack<var*>& step_assigned_vars) {
    v->value = value;
    v->assigned = true;

    unasigned_vars.erase(v);
    step_assigned_vars.push(v);
#ifdef DEBUG
    std::cout << "[ASSIGN " << (value ? "TRUE" : "FALSE" ) << " " << v->id << "]" << std::endl;
#endif
    //Pour toutes les formules qui contiennent v
    for(auto it = v->f.begin(); it != v->f.end(); ++it){
        if(it->second == value){ //On assigne true et la variable est positive dans la formule / on assigne false et la variable est négative
            it->first->actual_sum++;
        }

        if(it->first->actual_sum == it->first->max){ //Si on atteint la valeur max pour la formule
#ifdef DEBUG
            std::cout << "Maxed formula " << *(it->first) << std::endl;
#endif
            for(auto itv = it->first->vars.begin(); itv != it->first->vars.end(); ++itv){ //Alors pour toutes les variables de cette formule
#ifdef DEBUG
                std::cout << "\t" << itv->matching_var->id;
#endif
                if(!(*itv).matching_var->assigned){ //Qui n'ont pas déjà une valeur assignées
                    if(itv->positive) {
                        //On assigne la valeur false si elle ne sont pas négationner TODO Trouver le vrai mot
#ifdef DEBUG
                        std::cout << " to false" << std::endl;
#endif
                        assign_value(itv->matching_var, false, unasigned_vars, step_assigned_vars);
                    }
                    else{
                        //Ou la valeur true si de la forme (NOT A)
#ifdef DEBUG
                        std::cout << " to true" << std::endl;
#endif
                        assign_value(itv->matching_var, true, unasigned_vars, step_assigned_vars);
                    }
                }
            }
        }
    }
}

void Solver::remove_assign(Solver::var *v, std::set<var*>& unasigned_var) {

#ifdef DEBUG
    std::cout << "\033[32mRemove assign on " << v->id << "\033[0m" << std::endl;
#endif
    for(auto it = v->f.begin(); it != v->f.end(); ++it){
        if(it->second == v->value){
            it->first->actual_sum--;
        }
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
            os << itv->matching_var->id << " ";
        }
        os << std::endl;
    }
}

std::ostream &operator<<(std::ostream &os, const Solver::var &var) {
    os << "{ 'id': '" << var.id << "', 'assigned' : " << (var.assigned ? "'true' " : "'false'") << ", 'value' : "
       << (var.value ? "'true' " : "'false'") << ", 'f' : [ ";
    for (auto it = var.f.begin(); it != var.f.end(); ++it) {
        os << (*it).first->id << (it + 1 != var.f.end() ? " , " : "");
    }

    os << " ] }";
    return os;
}

std::ostream &operator<<(std::ostream &os, const Solver::formula &formula) {
    os << "{'id': " << formula.id << " 'min': " << formula.min << ", 'max' : " << formula.max << ", 'actual_sum' : "
       << formula.actual_sum << ", 'vars': [ ";
    for (auto it = formula.vars.begin(); it != formula.vars.end(); ++it) {
        os << "'" << (it->positive ? "" : "-") << it->matching_var->id << "'"
           << (it + 1 != formula.vars.end() ? " , " : "");
    }
    os << " ] }";
    return os;
}

std::ostream &Solver::display_var_value(std::ostream &os) {
    for(auto it = this->_var_table.begin(); it != this->_var_table.end(); ++it){
        os << it->first << " : " << (it->second->value ? "VRAI" : "FAUX") << std::endl;
    }

    return os;
}
