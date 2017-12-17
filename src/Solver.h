//
// Created by pviolette on 16/12/17.
//

#ifndef PROJECT_SOLVER_H
#define PROJECT_SOLVER_H


#include <vector>
#include <string>
#include <map>
#include <ostream>

class Solver {

public:
    struct formula_var;
    struct var;

    struct formula{
        int id;

        friend std::ostream &operator<<(std::ostream &os, const formula &formula);

        int min;
        int max;
        int actual_sum;
        std::vector<formula_var> vars;
    };

    struct formula_var{
        var * matching_var;
        bool positive;
    };

    struct var{
        std::string id;

        friend std::ostream &operator<<(std::ostream &os, const var &var);

        bool assigned;
        bool value;
        std::vector<std::pair<formula*, bool>> f;
    };

private:



public:

    std::map<std::string, var*> _var_table;

    std::vector<formula*> _formulas;

    Solver();
    virtual ~Solver();

    bool solve();
    void parseFile(const char * path);

    std::ostream& display_var_value(std::ostream& os);

    std::ostream& display_full(std::ostream &stream);

    std::ostream& display_readable_formula(std::ostream& os);

    void parseLine(std::string line);

    bool check_satisfy();

    void assign_value(var * v, bool value, std::set<var *>& unasigned_vars, std::stack<var*>& step_assigned_vars);

    void remove_assign(var * v, std::set<var*>& unasigned_var);
};


#endif //PROJECT_SOLVER_H
