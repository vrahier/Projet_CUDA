//
// Created by pviolette on 20/12/17.
//

#include "gpu_parse_file.h"

#include <vector>
#include <map>
#include "Utilities/file.h"

#include "gpu_defines.h"

struct tmp_formula {
    std::vector <std::pair<int, bool>> vars;
    int min;
    int max;

    friend ostream &operator<<(ostream &os, const tmp_formula &formula) {
        os << "{ min: " << formula.min << " max: " << formula.max << " vars : [ ";
        for (auto it = formula.vars.begin(); it != formula.vars.end(); ++it) {
            os << (it->second ? "" : "-") << it->first << " ";
        }
        os << "] }";
        return os;
    }
};


void
parseFile(const char *filepath, cume::Matrix<int> &dest, int &n_var, int &n_formulas) {
    std::map<std::string, int> name_id_table;
    std::vector <tmp_formula> formulas;

    n_var = 0;
    n_formulas = 0;

    File reader(filepath);

    for (auto it = reader.begin(); it != reader.end(); it++) {
        if (it->length() != 0) {
//            std::cout << "Parsing line \"" << *it << "\"" << std::endl;


            //Split de la formule
            std::istringstream iss(*it);

            std::vector <std::string> tokens;

            std::copy(std::istream_iterator<std::string>(iss),
                      std::istream_iterator<std::string>(),
                      std::back_inserter(tokens));
//            for (auto it = tokens.begin(); it != tokens.end(); ++it)
//                std::cout << *it << " | ";
//            std::cout << std::endl;
            tmp_formula f;
            //Premier et deuxième tokens : min et max de la formule
            f.min = std::stoi(tokens[0]);
            f.max = std::stoi(tokens[1]);

            //Traitement des variables
            for (auto itv = tokens.begin() + 2; itv != tokens.end(); ++itv) {

                bool p;
                std::string var_name;
                if ((*itv)[0] == '-') {
                    p = false;
                    var_name = itv->substr(1);
                } else {
                    p = true;
                    var_name = *itv;
                }

                int id;
                if (name_id_table.count(var_name) == 0) {
                    id = n_var;
                    n_var++;
                    name_id_table[var_name] = id;
                } else {
                    id = name_id_table[var_name];
                }

                f.vars.push_back(std::pair<int, bool>(id, p));
            }

            formulas.push_back(f);
        }
    }

    n_formulas = formulas.size();
/*
    for(auto it = name_id_table.begin(); it != name_id_table.end(); ++it){
        std::cout << "( " << it->first << " -> " << it->second << ") ";
    }
    std::cout << std::endl;
    for (auto it = formulas.begin(); it != formulas.end(); ++it) {
        std::cout << *it << std::endl;
    }
*/
    dest = cume::Matrix<int>(n_formulas, n_var + 2);

    std::memset(dest.m_cpu_data, 0, dest.size() * sizeof(int));

    for (int i = 0; i < n_formulas; ++i) {
        int offset = i * (n_var + 2);
        dest[offset + F_MIN] = formulas[i].min;
        dest[offset + F_MAX] = formulas[i].max;


        for (auto it = formulas[i].vars.begin(); it != formulas[i].vars.end(); ++it) {
            int value;
            if (it->second) {
                value = 1;
            } else {
                value = -1;
            }
            dest[offset + 2 + it->first] = value;
        }
    }

}
