#include "file.h"
#include <iostream>
#include <fstream>


void File::loadFile(const char* path)
{
  std::ifstream input(path, std::ifstream::in);
  if((m_opened = input.is_open())){
    while(!input.eof()){
      std::string str;
      std::getline(input, str);
      m_content.push_back(str);
    }
    input.close();
  }
  else{
   std::cerr << "[FILE]Impossible d'ouvrir le fichier " << path << std::endl; 
  }
}

File::File(const char* path)
{
  loadFile(path);
}

File::File(const std::string& path)
{
  loadFile(path.c_str());
}

const std::string& File::getLine(unsigned int line) const
{
  return m_content[line];
}

std::string& File::getLine(unsigned int line)
{
  return m_content[line];
}


void File::save(const std::string& path)
{
  std::ofstream out(path.c_str(), std::ofstream::out);
  if(out.is_open()){
    for(unsigned int i = 0; i < m_content.size(); i++){
      out << m_content[i] << std::endl;
    }
    out.close();
  }
  else{
   std::cerr << "Impossible d'ouvrir le fichier " << path << std::endl; 
  }
}

std::vector< std::string >::iterator File::begin()
{
  return m_content.begin();
}

std::vector< std::string >::const_iterator File::begin() const
{
  return m_content.begin();
}

std::vector< std::string >::iterator File::end()
{
  return m_content.end();
}

std::vector< std::string >::const_iterator File::end() const
{
  return m_content.end();
}

unsigned int File::lineNumber() const
{
  return (unsigned int) m_content.size();
}

bool File::open()
{
  return m_opened;
}


