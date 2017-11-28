/* VIOLETTE Paulin
 * L3 Informatique - UFR Sciences Angers
 * file.h
 * 
 * version 0.1
 */

#ifndef FILE_H
#define FILE_H

#include <vector>
#include <string>

/* Class File
 * 
 * Lit un fichier texte et stocke le contenu du fichier dans un vector de string
 */
class File
{
private:
  std::vector<std::string> m_content;
  bool m_opened;
  
  void loadFile(const char* path);
public:
  File(const char* path);
  File(const std::string & path);
  
  /*
   * Sauvegarde le contenu dans le fichier path
   */
  void save(const std::string & path);
  
  /*
   * Renvoie la ligne i (lignes indexé à partir de 0)
   */
  const std::string& getLine(unsigned int line) const;
  std::string& getLine(unsigned int line);
 
  /* Function d'accès pour le iterator sur le vector content
   * 
   */
  std::vector<std::string>::const_iterator begin() const;
  std::vector<std::string>::const_iterator end() const;
  std::vector<std::string>::iterator begin();
  std::vector<std::string>::iterator end();
  
  /*
   * Renvoie le nombre de ligne dans le fichier
   */
  unsigned int lineNumber() const;
  
  /*
   * Vrai si le fichier a été correctement ouvert
   */
  bool open();
};

#endif // FILE
