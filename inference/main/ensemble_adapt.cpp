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

#ifdef SEAL
const bool use_seal = true;
#else
const bool use_seal = false;
#endif

vector<double> logit(const Tensor<3,Scalar>& decoded_output);
tuple<vector<string>,string,size_t,double,vector<int>,vector<int>,size_t> parse_arguments(int argc, char* argv[]);

int main(int argc, char* argv[])
try
{
    // check if model and input files are provided
    if (argc < 7)
    {
        cout << "Usage: " << argv[0] << " <model_1.json> ... <model_M.json> <input.json> |N| |scale| |q0|,|q1|,...,|qL|,|P| <rotation_steps> [# iterations]" << endl;
        cout << "Example: " << argv[0] << " model_1.json model_2.json input.json 15 22 1x54,1x23,18x44 1,2,4,8,16,24,31,32,64,-1,-2,-4,-8,-16,-32" << endl;
        return 1;
    }

    auto [filename_models, filename_io, n, scale, logqi, rotation_steps, niters] = parse_arguments(argc, argv);
    bool show_rotation = !use_seal && rotation_steps.size() == 1 && rotation_steps[0] == 0;

    // Initializing keys
    shared_ptr<Keys> keys_ptr = make_shared<Keys>(Keys(n, logqi, scale, rotation_steps)); // slots = n / 2
    Plaintext::default_keys(keys_ptr);
    Ciphertext::default_keys(keys_ptr);

    cout << "Loading input "  << filename_io << " ... " << flush;
    nlohmann::json io;
    {
        ifstream fin(filename_io);
        fin >> io;
    }
    cout << "ok" << endl;

    cout << "Loading models " << filename_models << " ... " << flush;
    hw::Ensemble model(filename_models, shape(io[0]["input"].get<vector<vector<vector<Scalar>>>>()));
    cout << "ok" << endl;

    cout << "Model:" << endl;
    model.print();

    time_point<high_resolution_clock> start, end;
    size_t correct = 0;
    size_t nsamples = niters ? niters : io.size();
    for (size_t i = 0; i < nsamples; i++)
    {
        keys_ptr->reset_rotation();
        auto& test_data = io[i]; // sample

        // encoding
        auto input = model.encode_input(Tensor<3,Scalar>::copy(test_data["input"].get<vector<vector<vector<Scalar>>>>()));

        // forward
        start = high_resolution_clock::now();
        auto yhat = model.forward(input);
        end = high_resolution_clock::now();
        auto time_span = double(duration_cast<milliseconds>(end - start).count()) / 1000;

        // decoding
        auto output = model.decode_output(yhat);

        // classifying
        auto probabilities = model.classify(output);
        auto y_hat = probabilities.argmax();
        auto y = test_data["output"].get<vector<Scalar>>()[0];
        if (y_hat == y) correct++;

        streamsize default_precision = cout.precision();
        cout << "Sample " << i << ", Predicted: " << y_hat << ", Expected: " << y << ", Correct: " << (y_hat == y) << endl;
        cout << "Output:" << endl;
        for (const auto& model_output : output) cout << logit(model_output) << endl;
        cout << "Probabilities: " << probabilities << endl;
        cout << "Accuracy: " << correct / double(i+1) << fixed << setprecision(3) << " Inference time: " << time_span << "s" << endl;
        cout.unsetf(ios::fixed);
        cout.precision(default_precision);
        
        if (show_rotation) keys_ptr->print_rotation();
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

tuple<vector<string>,string,size_t,double,vector<int>,vector<int>,size_t> parse_arguments(int argc, char* argv[])
{
    // check if # iterations is informed
    size_t niters = 0UL;
    int offset = -5;
    try
    {
        string last = argv[argc - 1];
        if (!last.empty() && last.find_first_not_of("0123456789") == string::npos)
        {
            niters = stoul(argv[argc - 1]);
            offset--;
        }
    }
    catch(...) {}
    
    vector<string> filename_models; // a vector of models for ensemble
    for (int i = 1; i < argc + offset; i++) filename_models.push_back(argv[i]);

    string filename_io = argv[argc + offset]; // dataset for inference
    
    size_t n; // polynomial degree
    try { n = 1UL << stoul(argv[argc + offset + 1]); }
    catch(...) { throw "Invalid log(N)"; }

    double scale;
    try { scale = double(1ULL << stoull(argv[argc + offset + 2])); }
    catch(...) { throw "Invalid scaling factor"; }

    vector<int> logqis;
    try
    {
        for (const string& item : split(argv[argc + offset + 3], ','))
        {
            vector<string> tokens = split(item, 'x');
            int occurrences = stoi(tokens[0]);
            int logqi = stoi(tokens[1]);
            while (occurrences-- > 0) logqis.push_back(logqi);
        }
    }
    catch (...) { throw "Invalid sizes of RNS moduli"; }

    vector<int> rotation_steps;
    try { for (const string& rotation_step : split(argv[argc + offset + 4], ',')) rotation_steps.push_back(stoi(rotation_step)); }
    catch (...) { throw "Invalid rotation steps"; }

    return make_tuple(filename_models, filename_io, n, scale, logqis, rotation_steps, niters);
}
