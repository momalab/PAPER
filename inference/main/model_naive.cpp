#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <vector>
#include "encode.h"
#include "ml.h"
#include "scalar.h"
#include "tensor.h"
#include "text.h"
#include "vectorization.h"

using namespace fhe;
using namespace ml;
using namespace std;
using namespace type;
using namespace util;
using namespace std::chrono;

vector<double> logit(const Tensor<3,Scalar>& decoded_output);
tuple<string,string,size_t,double,vector<int>,size_t> parse_arguments(int argc, char* argv[]);

int main(int argc, char* argv[])
try
{
    // check if model and input files are provided
    if (argc < 6)
    {
        cout << "Usage: " << argv[0] << " <model.json> <input.json> |N| |scale| |q0|,|q1|,...,|qL|,|P| [# iterations]" << endl;
        cout << "Example: " << argv[0] << " model.json input.json 15 22 1x54,1x23,18x44" << endl;
        return 1;
    }

    auto [filename_model, filename_io, n, scale, logqi, niters] = parse_arguments(argc, argv);

    // Initializing keys
    shared_ptr<Keys> keys_ptr = make_shared<Keys>(Keys(n, logqi, scale));
    Plaintext::default_keys(keys_ptr);
    Ciphertext::default_keys(keys_ptr);

    cout << "Loading input ... " << flush;
    nlohmann::json io;
    {
        ifstream fin(filename_io);
        fin >> io;
    }
    cout << "ok" << endl;

    cout << "Loading model ... " << flush;
    naive::Model model(filename_model, shape(io[0]["input"].get<vector<vector<vector<Scalar>>>>()));
    cout << "ok" << endl;

    cout << "Model:" << endl;
    model.print();

    size_t correct = 0;
    size_t nsamples = niters ? niters : io.size();
    for (size_t i = 0; i < nsamples; i++)
    {
        auto& test_data = io[i];

        // encoding
        auto input = model.encode_input(Tensor<3,Scalar>::copy(test_data["input"].get<vector<vector<vector<Scalar>>>>()));

        // forward and decoding
        auto output = model.decode_output(model.forward(input));
        // auto output = model.decode_output(model.forward(input))[0][0];

        // classifying
        auto probabilities = model.classify(output);
        auto y_hat = probabilities.argmax();
        auto y = test_data["output"].get<vector<Scalar>>()[0];
        if (y_hat == y) correct++;

        cout << "Sample " << i << ", Predicted: " << y_hat << ", Expected: " << y << ", Correct: " << (y_hat == y) << endl;
        cout << "Output:" << endl;
        cout << logit(output) << endl;
        cout << "Probabilities: " << probabilities << endl;
        cout << "Accuracy: " << correct / double(i+1) << endl;

        // auto y_hat = output.argmax();
        // auto y = test_data["output"].get<vector<Scalar>>()[0];
        // if (y_hat == y) correct++;
        
        // cout << "Sample " << i << ", Predicted: " << y_hat << ", Expected: " << y << ", Correct: " << (y_hat == y) << ", " << flush;
        // cout << "Output: " << output << endl;
        // cout << "Accuracy: " << correct / double(i+1) << endl;
    }
    cout << "Accuracy: " << correct / double(nsamples) << endl;
}
catch (const char* e) { cerr << e << endl; }
catch (const string& e) { cerr << e << endl; }
catch (const nlohmann::detail::parse_error& e) { cerr << "Parse error: " << e.what() << endl; }
catch (const nlohmann::detail::type_error& e) { cerr << "Type error: " << e.what() << endl; }
catch (const nlohmann::detail::exception& e) { cerr << "Exception: " << e.what() << endl; }
catch (const exception& e) { cerr << "Exception: " << e.what() << endl; }

vector<double> logit(const Tensor<3,Scalar>& decoded_output)
{
    vector<double> logit;
    for (const auto& channel : decoded_output) logit.push_back(channel[0][0]);
    return logit;
}

tuple<string,string,size_t,double,vector<int>,size_t> parse_arguments(int argc, char* argv[])
{
    if (argc < 6) throw "Not enough arguments to parse";

    string filename_model = argv[1]; // model

    string filename_io = argv[2]; // dataset for inference
    
    size_t n; // polynomial degree
    try { n = 1UL << stoul(argv[3]); }
    catch(...) { throw "Invalid log(N)"; }

    double scale;
    try { scale = double(1ULL << stoull(argv[4])); }
    catch(...) { throw "Invalid scaling factor"; }

    vector<int> logqis;
    try
    {
        for (const string& item : split(argv[5], ','))
        {
            vector<string> tokens = split(item, 'x');
            int occurrences = stoi(tokens[0]);
            int logqi = stoi(tokens[1]);
            while (occurrences-- > 0) logqis.push_back(logqi);
        }
    }
    catch (...) { throw "Invalid sizes of RNS moduli"; }

    size_t niters = argc >= 7 ? stoul(argv[6]) : 0UL;

    return make_tuple(filename_model, filename_io, n, scale, logqis, niters);
}
