#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include "ml.h"
#include "scalar.h"
#include "tensor.h"
#include "vectorization.h"

using namespace fhe;
using namespace ml;
using namespace std;
using namespace type;
using namespace util;

vector<double> logit(const Tensor<3,Scalar>& decoded_output, size_t model_idx);
tuple<vector<string>,string,size_t> parse_arguments(int argc, char* argv[]);

int main(int argc, char* argv[])
try
{
    // check if model and input files are provided
    if (argc < 3)
    {
        cout << "Usage: " << argv[0] << " <model_1.json> ... <model_M.json> <input.json> [# iterations]" << endl;
        return 1;
    }

    auto [filename_models, filename_io, niters] = parse_arguments(argc, argv);

    cout << "Loading input "  << filename_io << " ... " << flush;
    nlohmann::json io;
    {
        ifstream fin(filename_io);
        fin >> io;
    }
    cout << "ok" << endl;

    cout << "Loading models " << filename_models << " ... " << flush;
    plain::Ensemble model(filename_models, shape(io[0]["input"].get<vector<vector<vector<Scalar>>>>()));
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

        // classifying
        auto probabilities = model.classify(output);
        auto y_hat = probabilities.argmax();
        auto y = test_data["output"].get<vector<Scalar>>()[0];
        if (y_hat == y) correct++;
        
        cout << "Sample " << i << ", Predicted: " << y_hat << ", Expected: " << y << ", Correct: " << (y_hat == y) << endl;
        cout << "Output:" << endl;
        for (size_t model_idx = 0; model_idx < output[0][0].size(); model_idx++) cout << logit(output, model_idx) << endl;
        cout << "Probabilities: " << probabilities << endl;
        cout << "Accuracy: " << correct / double(i+1) << endl;
    }
    cout << "Accuracy: " << correct / double(nsamples) << endl;
}
catch (const char* e) { cerr << e << endl; }
catch (const string& e) { cerr << e << endl; }
catch (const nlohmann::detail::parse_error& e) { cerr << "Parse error: " << e.what() << endl; }
catch (const nlohmann::detail::type_error& e) { cerr << "Type error: " << e.what() << endl; }
catch (const nlohmann::detail::exception& e) { cerr << "Exception: " << e.what() << endl; }
catch (const exception& e) { cerr << "Exception: " << e.what() << endl; }

vector<double> logit(const Tensor<3,Scalar>& decoded_output, size_t model_idx)
{
    vector<double> logit;
    for (const auto& channel : decoded_output) logit.push_back(channel[0][model_idx]);
    return logit;
}

tuple<vector<string>,string,size_t> parse_arguments(int argc, char* argv[])
{
    // check if # iterations is informed
    size_t niters = 0UL;
    int offset = -1;
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
    
    return make_tuple(filename_models, filename_io, niters);
}
