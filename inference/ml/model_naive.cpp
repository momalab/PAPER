#include "model_naive.h"

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_set>
#include <vector>
#include "ciphertext.h"
#include "encode.h"
#include "layer.h"
#include "math.h"
#include "ml_naive.h"
#include "model_base.h"
#include "tensor.h"

using namespace ml;
using namespace std;
using namespace type;

namespace fhe
{

namespace naive
{

// Constructors

Model::Model(const nlohmann::json& json, const vector<size_t>& ishape) : BaseModel(json, ishape) {}

Model::Model(ifstream is, const vector<size_t>& ishape) : BaseModel(move(is), ishape) {}

Model::Model(const string& filename, const vector<size_t>& ishape) : BaseModel(filename, ishape) {}

// Private functions

void Model::compute(size_t node_id, std::unordered_set<size_t>& visited, unordered_map<size_t,Tensor<3,Ciphertext>>& inouts) const
{
    const Node& node = nodes[node_id - 1];

    // compute unvisited dependencies
    for (size_t previous_id : node.previous)
        if (previous_id && visited.find(previous_id) == visited.end())
            compute(previous_id, visited, inouts);

    // compute node
    const auto& layer = node.layer;
    const auto& kernel = layer.kernel();
    const auto& bias = layer.bias();
    const auto& kshape = layer.kshape();
    const auto& stride = layer.stride();
    const auto& padding = layer.padding();
    const auto& divisor = layer.divisor();
    const auto& input = inouts[node.previous[0]];
    switch (layer.type())
    {
        case LayerType::ADD: inouts[node_id] = add(input, inouts[node.previous[1]]); break;
        case LayerType::AVGPOOL2D: inouts[node_id] = avgpool2d(input, kshape, stride, padding, divisor); break;
        case LayerType::BATCHNORM2D: inouts[node_id] = batchnorm2d(input, get<Tensor<4,Scalar>>(kernel)); break;
        case LayerType::CONV2D: inouts[node_id] = conv2d(input, get<Tensor<6,Scalar>>(kernel), bias, stride, padding); break;
        case LayerType::LINEAR: inouts[node_id] = linear(input, get<Tensor<4,Scalar>>(kernel), bias); break;
        case LayerType::POLY: inouts[node_id] = poly(input, get<Tensor<4,Scalar>>(kernel)); break;
        case LayerType::POLYSKIP: inouts[node_id] = polyskip(input, inouts[node.previous[1]], get<Tensor<4,Scalar>>(kernel)); break;
        default: throw invalid_argument("Model::compute: LayerType not implemented");
    }

    // mark node as visited
    visited.insert(node_id);

    // destroy previous node outputs if their next nodes have been visited (to reduce peak memory)
    for (size_t previous_id : node.previous) if (previous_id)
    {
        const Node& previous_node = nodes[previous_id - 1];
        if (all_of(previous_node.next.begin(), previous_node.next.end(), [&visited](size_t id) { return visited.find(id) != visited.end(); }))
            inouts.erase(previous_id);
    }
}

// Public functions

Tensor<1,double> Model::classify(const type::Tensor<3,Scalar>& decoded_output) const
{
    Tensor<1,double> r;
    vector<double> logit;
    for (const auto& channel : decoded_output) logit.push_back(channel[0][0]);
    r.vector() = util::softmax(logit);
    return r;
}

Tensor<3,Scalar> Model::decode_output(const Tensor<3,Ciphertext>& output) const
{
    return fhe::naive::decode_output(output);
}

Tensor<3,Ciphertext> Model::encode_input(const Tensor<3,Scalar>& input) const
{
    return fhe::naive::encode_input(input);
}

Encoding Model::encoding() const
{
    return Encoding::NAIVE;
}

Tensor<3,Ciphertext> Model::forward(Tensor<3,Ciphertext>& input) const
{
    unordered_set<size_t> visited;
    unordered_map<size_t,Tensor<3,Ciphertext>> inouts;
    inouts[0] = move(input);
    compute(last_id, visited, inouts);
    input = move(inouts[0]);
    return inouts[last_id];
}

Tensor<3,Ciphertext> Model::forward(const Tensor<3,Ciphertext>& input) const
{
    Tensor<3,Ciphertext> input_copy = input;
    return forward(input_copy);
}

void Model::print() const
{
    return BaseModel::print();
}

} // naive

} // fhe