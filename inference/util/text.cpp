#include "text.h"

#include <sstream>
#include <string>
#include <vector>

using namespace std;

namespace util
{

vector<string> split(const string& s, char delimiter)
{
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) tokens.push_back(token);
    return tokens;
}

} // util