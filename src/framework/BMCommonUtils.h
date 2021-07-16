#ifndef BMCOMMONUTILS_H
#define BMCOMMONUTILS_H

#include <set>
#include <string>
#include <sstream>

namespace bm {

class Uncopiable {
    public:
    Uncopiable() = default;
    Uncopiable(Uncopiable&) = delete;
    Uncopiable(const Uncopiable&) = delete;
    Uncopiable& operator = (const Uncopiable) = delete;
};

template <typename T>
std::set<T> stringToSet(const std::string &s)
{
    std::set<T> values;
    std::string new_str = s;
    for(auto& c: new_str){
       if(!isdigit(c) && c!='.') c= ' ';
    }
    std::istringstream is(new_str);
    T v= 0;
    while(is>>v){
        values.insert(v);
    }
    return values;
}

}

#endif // BMCOMMONUTILS_H
