#include "ensemble_plain.h"

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_set>
#include <vector>
#include "encodings.h"
#include "layer.h"
#include "math.h"
#include "ml_plain.h"
#include "model_base.h"
#include "scalar.h"
#include "tensor.h"
#include "vectorization.h"

using namespace ml;
using namespace std;
using namespace type;

namespace fhe
{

namespace plain
{

// Constructors

Ensemble::Ensemble(const vector<nlohmann::json>& jsons, const vector<size_t>& ishape)
{
    for (const auto& json : jsons) models.emplace_back(Model(json, ishape));
}

Ensemble::Ensemble(vector<ifstream> iss, const vector<size_t>& ishape)
{
    for (auto& is : iss) models.emplace_back(Model(move(is), ishape));
}


Ensemble::Ensemble(const vector<string>& filenames, const vector<size_t>& ishape)
{
    for (const auto& filename : filenames) models.emplace_back(Model(filename, ishape));
}

// Public functions

Tensor<1,double> Ensemble::classify(const Tensor<3,Scalar>& decoded_output) const
{
    if (models.empty()) throw runtime_error("Ensemble not initialized.");
    size_t n_models = models.size();
    size_t n_classes = decoded_output.shape()[0];
    Tensor<1,double> aggregated_probabilities{n_classes};
    
     // Average the probabilities across models
    for (size_t model_idx = 0; model_idx < n_models; model_idx++)
    {
        vector<double> logit;
        for (size_t i = 0; i < n_classes; i++) logit.push_back(decoded_output[i][0][model_idx]);
        auto model_probabilities = util::softmax(logit);
        util::operator+=(aggregated_probabilities.vector(), model_probabilities); // Accumulate probabilities
    }
    for (auto& prob : aggregated_probabilities.vector()) prob /= double(n_models);

    return aggregated_probabilities;
}

Tensor<3,Scalar> Ensemble::decode_output(const Tensor<3,Scalar>& output) const
{
    return output;
}

Tensor<3,Scalar> Ensemble::encode_input(const Tensor<3,Scalar>& input) const
{
    return input;
}

Encoding Ensemble::encoding() const
{
    return Encoding::PLAIN;
}

Tensor<3,Scalar> Ensemble::forward(const Tensor<3,Scalar>& input) const
{
    if (models.empty()) throw runtime_error("No models in the ensemble.");

    auto prediction = models.front().forward(input);
    auto shape = prediction.shape();
    for (size_t idx = 1; idx < models.size(); idx++)
    {
        auto model_output = models[idx].forward(input);
        for (size_t i = 0; i < shape[0]; i++)
        {
            for (size_t j = 0; j < shape[1]; j++)
            {
                auto& v1 = prediction[i][j].vector();
                auto& v2 = model_output[i][j].vector();
                v1.insert(v1.end(), v2.begin(), v2.end());
            }
        }
    }
    return prediction;
}

void Ensemble::print() const
{
    if (!models.empty()) models.front().print();
}

} // plain

} // fhe