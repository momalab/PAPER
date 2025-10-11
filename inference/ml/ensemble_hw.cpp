#include "ensemble_hw.h"

#include <algorithm>
#include <fstream>
#include <chrono>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>
#include "ciphertext.h"
#include "defines.h"
#include "encode.h"
#include "layer.h"
#include "math.h"
#include "ml_hw.h"
#include "model_base.h"
#include "tensor.h"
#include "vectorization.h"

using namespace ml;
using namespace std;
using namespace std::chrono;
using namespace type;

namespace fhe
{

namespace hw
{

// Constructors

Ensemble::Ensemble(const vector<nlohmann::json>& jsons, const vector<size_t>& ishape)
{
    for (const auto& json : jsons) models.emplace_back(json, ishape);
    init();
}

Ensemble::Ensemble(vector<ifstream> iss, const vector<size_t>& ishape)
{
    for (auto& is : iss) models.emplace_back(move(is), ishape);
    init();
}

Ensemble::Ensemble(const vector<string>& filenames, const vector<size_t>& ishape)
{
    for (const auto& filename : filenames) models.emplace_back(filename, ishape);
    init();
}

// Private functions

void Ensemble::compute(size_t node_id, std::unordered_set<size_t>& visited, unordered_map<size_t,shared_ptr<Tensor<2,Ciphertext>>>& inouts) const
{
    size_t node_idx = node_id - 1;
    const auto& nodes = models[0].get_nodes();
    const auto& node = nodes[node_idx];

    // compute unvisited dependencies
    for (size_t previous_id : node.previous)
        if (previous_id && visited.find(previous_id) == visited.end())
            compute(previous_id, visited, inouts);

    time_point<high_resolution_clock> start, end;
    start = high_resolution_clock::now();

    // check if computation can be done in place
    bool inplace = true;
    for (size_t previous_id : node.previous)
    {
        if (!previous_id){ inplace = false; break; }
        for (size_t next_id : nodes[previous_id - 1].next) // check if all next nodes apart from node_id have been visited
            if (next_id != node_id && visited.find(next_id) == visited.end()) { inplace = false; break; }
    }
    
    // compute node
    auto& layer = node.layer;
    auto& kernel = kernels[node_idx];
    auto& bias = biases[node_idx];
    auto& config = configs[node_idx];
    auto input_ptr = inouts[node.previous[0]];
    auto& input = *input_ptr;
    size_t nthreads = config::NTHREADS;
    if (layer.type() != LayerType::ADD  && layer.type() != LayerType::BATCHNORM2D &&
        layer.type() != LayerType::POLY && layer.type() != LayerType::POLYSKIP) inplace = false;

    std::cout << "Node " << node_id << " " << to_string(node.layer.type(), true) << "\tInplace: " << inplace << " " << node.previous << flush;

    if (inplace)
    {
        switch (layer.type())
        {
            case LayerType::ADD: add_inplace(input, *inouts[node.previous[1]], config, configs[node.previous[1] - 1], nthreads, offsets); break;
            case LayerType::BATCHNORM2D: batchnorm2d_inplace(input, get<Tensor<5,Scalar>>(kernel), config, nthreads, offsets); break;
            case LayerType::POLY: poly_inplace(input, get<Tensor<5,Scalar>>(kernel), config, nthreads, offsets); break;
            default: throw invalid_argument("Model::compute: LayerType not implemented");
        }
        inouts[node_id] = input_ptr;
    }
    else 
    {
        if (!inouts[node_id]) inouts[node_id] = make_shared<Tensor<2,Ciphertext>>();
        switch (layer.type())
        {
            case LayerType::ADD: *inouts[node_id] = add(input, *inouts[node.previous[1]], config, configs[node.previous[1] - 1], nthreads); break;
            case LayerType::AVGPOOL2D: *inouts[node_id] = avgpool2d(input, config, nthreads); break;
            case LayerType::BATCHNORM2D: *inouts[node_id] = batchnorm2d(input, get<Tensor<5,Scalar>>(kernel), config, nthreads, offsets); break;
            case LayerType::CONV2D: *inouts[node_id] = conv2d(input, get<Tensor<7,Scalar>>(kernel), bias, config, nthreads, offsets); break;
            case LayerType::LINEAR: *inouts[node_id] = linear(input, get<Tensor<5,Scalar>>(kernel), bias, config, nthreads, offsets); break;
            case LayerType::POLY: *inouts[node_id] = poly(input, get<Tensor<5,Scalar>>(kernel), config, nthreads, offsets); break;
            default: throw invalid_argument("Model::compute: LayerType not implemented");
        }
    }

    // mark node as visited
    visited.insert(node_id);

    // destroy previous node outputs if their next nodes have been visited (to reduce peak memory)
    for (size_t previous_id : node.previous) if (previous_id)
    {
        const auto& previous_node = nodes[previous_id - 1];
        if (all_of(previous_node.next.begin(), previous_node.next.end(), [&visited](size_t id) { return visited.find(id) != visited.end(); }))
            inouts.erase(previous_id);
    }
    end = high_resolution_clock::now();
    auto time_span = double(duration_cast<milliseconds>(end - start).count()) / 1000;

    std::cout << "\t(in " << time_span << "s) level: " << (*inouts[node_id])[0][0].level() << " scale: " << (*inouts[node_id])[0][0].scale() << std::endl;
}

void Ensemble::config(size_t node_id, unordered_set<size_t>& visited) // create configurations from layers
{
    size_t node_idx = node_id - 1;
    const auto& node = models[0].get_nodes()[node_idx];

    // config unvisited dependencies
    for (size_t previous_id : node.previous)
        if (previous_id && visited.find(previous_id) == visited.end())
            config(previous_id, visited);
    
    // config node
    size_t previous_id = node.previous[0]; // FIXME it should be the deepest path
    auto& config = configs[node_idx];

    const auto& oshape = node.layer.oshape();
    const auto& ci = oshape[0];
    Layer dummy_layer(LayerType::AVGPOOL2D, Tensor<4,Scalar>(), Tensor<3,Scalar>(), oshape, {ci, ci, 1, 1}, vector<size_t>(), {1,1}, {0,0}, 1.0);

    if (previous_id)
    {
        const auto& prev_config = configs[previous_id - 1];
        config = Configuration({node.layer, dummy_layer}, prev_config.omap(), prev_config.nct());
    }
    else
    {
        vector<size_t> max_padding = {0, 0};
        config = Configuration({node.layer, dummy_layer}, max_padding);
    }

    config.next(); // position head
    
    // set first node
    for (size_t previous_id : node.previous)
        if (!previous_id) first_id = node_id;

    // compute multiplicative depth
    int previous_depth = 0;
    for (size_t previous_id : node.previous)
    {
        if (!previous_id) continue; // skip input
        previous_depth = max(previous_depth, out_depths[previous_id - 1]);
    }
    const auto& type = node.layer.type();
    if (type == LayerType::POLY || type == LayerType::POLYSKIP)
    {
        int remainder = previous_depth % fhe::SUBLEVELS;
        int refit = (remainder ? fhe::SUBLEVELS - remainder : 0);
        previous_depth += refit;
    }
    out_depths[node_idx] = in_depths[node_idx] = previous_depth;
    if (type == LayerType::AVGPOOL2D) out_depths[node_idx] += int(node.layer.divisor() != 1); // add 1 if divisor is not 1
    else if (type != LayerType::ADD) out_depths[node_idx]++; // add 1 for the layer itself

    // mark node as visited
    visited.insert(node_id);
}

void Ensemble::init()
{
    if (models.empty()) throw invalid_argument("Model::init: No models provided");
    this->last_id = models[0].get_last_id();
    size_t nNodes = models[0].get_nodes().size();
    
    configs.resize(nNodes);
    in_depths.resize(nNodes);
    out_depths.resize(nNodes);
    unordered_set<size_t> visited;
    config(last_id, visited);

    // compute offsets for ensemble
    const auto& config = configs[first_id - 1];
    const auto& mapping = config.mapping();
    const auto& nModels = models.size();
    auto ensemble_slotsize = 1UL << (util::ceil_log2(mapping.back().back()) + 1); // +1 to leave empty space between models
    offsets.resize(nModels);
    for (size_t i = 0; i < nModels; i++) offsets[i] = int(ensemble_slotsize * i); // offsets for each model in the ensemble

    // move kernels and biases to the ensemble
    kernels.resize(nNodes);
    biases.resize(nNodes);
    for (size_t node_idx = 0; node_idx < nNodes; node_idx++)
    {
        auto type = models[0].get_nodes()[node_idx].layer.type();
        if (type == LayerType::CONV2D) kernels[node_idx] = Tensor<7,Scalar>();
        else kernels[node_idx] = Tensor<5,Scalar>();
        for (size_t model_idx = 0; model_idx < nModels; model_idx++)
        {
            auto& model = models[model_idx];
            auto& node = model.get_nodes()[node_idx];

            if (type == LayerType::CONV2D)
            {
                auto& this_kernel = get<Tensor<7,Scalar>>(kernels[node_idx]);
                auto& model_kernel = get<Tensor<6,Scalar>>(node.layer.kernel());
                this_kernel.emplace_back(model_kernel);
            }
            else
            {
                auto& this_kernel = get<Tensor<5,Scalar>>(kernels[node_idx]);
                auto& model_kernel = get<Tensor<4,Scalar>>(node.layer.kernel());
                this_kernel.emplace_back(model_kernel);
            }

            auto& model_bias = node.layer.bias();
            biases[node_idx].emplace_back(model_bias);
        }
    }
}

// Public functions

type::Tensor<1,double> Ensemble::classify(const type::Tensor<4,Scalar>& decoded_output) const
{
    if (models.empty()) throw runtime_error("Ensemble::classify: Ensemble not initialized.");

    size_t nModels = models.size();
    size_t nClasses = decoded_output.shape()[1];

    // Average the probabilities across models
    Tensor<1,double> aggregated_probabilities{nClasses};
    for (size_t model_idx = 0; model_idx < nModels; model_idx++)
    {
        vector<double> logit;
        for (size_t i = 0; i < nClasses; i++) logit.push_back(decoded_output[model_idx][i][0][0]);
        auto model_probabilities = util::softmax(logit);
        util::operator+=(aggregated_probabilities.vector(), model_probabilities); // Accumulate probabilities
    }
    aggregated_probabilities *= 1.0 / double(nModels); // Normalize the probabilities

    return aggregated_probabilities;
}

Tensor<4,Scalar> Ensemble::decode_output(const Tensor<2,Ciphertext>& output) const
{
    const auto& config = configs[last_id - 1];
    return fhe::hw::decode_output(output, config, config::NTHREADS, offsets);
}

Tensor<2,Ciphertext> Ensemble::encode_input(const Tensor<3,Scalar>& input) const
{
    auto config = configs[first_id - 1];
    config.reset(); // move head to beginning and encode input
    Tensor<4,Scalar> x;
    for (size_t model_idx = 0; model_idx < models.size(); model_idx++) x.emplace_back(input);
    return fhe::hw::encode_input(x, config, config::NTHREADS, -1, 0.0, offsets);
}

Encoding Ensemble::encoding() const
{
    return Encoding::HW;
}

Tensor<2,Ciphertext> Ensemble::forward(const Tensor<2,Ciphertext>& input) const
{
    if (models.empty()) throw invalid_argument("Ensemble::forward: There are no models in the ensemble");

    unordered_set<size_t> visited;
    unordered_map<size_t,shared_ptr<Tensor<2,Ciphertext>>> inouts;
    inouts[0] = make_shared<Tensor<2,Ciphertext>>(input);
    compute(last_id, visited, inouts);
    return *inouts[last_id];
}

void Ensemble::print() const
{
    if (models.empty()) throw runtime_error("Ensemble::print: Ensemble not initialized.");
    return models[0].print();
}

} // hw

} // fhe
