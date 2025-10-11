#include "model_hw.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
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

Model::Model(const nlohmann::json& json, const vector<size_t>& ishape) : BaseModel(json, ishape) { init(); }

Model::Model(ifstream is, const vector<size_t>& ishape) : BaseModel(move(is), ishape) { init(); }

Model::Model(const string& filename, const vector<size_t>& ishape) : BaseModel(filename, ishape) { init(); }

// Private functions

void Model::compute(size_t node_id, std::unordered_set<size_t>& visited, unordered_map<size_t,shared_ptr<Tensor<2,Ciphertext>>>& inouts) const
{
    size_t node_idx = node_id - 1;
    const Node& node = nodes[node_idx];

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
    auto& config = configs[node_idx];
    auto input_ptr = inouts[node.previous[0]];
    auto& input = *input_ptr;
    if (layer.type() != LayerType::ADD  && layer.type() != LayerType::BATCHNORM2D &&
        layer.type() != LayerType::POLY && layer.type() != LayerType::POLYSKIP) inplace = false;

    std::cout << "Node " << node_id << " " << to_string(node.layer.type(), true) << "\tInplace: " << inplace << " " << node.previous << flush;

    if (inplace)
    {
        switch (layer.type())
        {
            case LayerType::ADD: add_inplace(input, *inouts[node.previous[1]], config, configs[node.previous[1] - 1]); break;
            case LayerType::BATCHNORM2D: batchnorm2d_inplace(input, get<Tensor<4,Scalar>>(node.layer.kernel()), config); break;
            case LayerType::POLY: poly_inplace(input, get<Tensor<4,Scalar>>(node.layer.kernel()), config); break;
            case LayerType::POLYSKIP: polyskip_inplace(input, *inouts[node.previous[1]], get<Tensor<4,Scalar>>(node.layer.kernel()), config, configs[node.previous[1] - 1]); break;
            default: throw invalid_argument("Model::compute: LayerType not implemented");
        }
        inouts[node_id] = input_ptr;
    }
    else 
    {
        if (!inouts[node_id]) inouts[node_id] = make_shared<Tensor<2,Ciphertext>>();
        switch (layer.type())
        {
            case LayerType::ADD: *inouts[node_id] = add(input, *inouts[node.previous[1]], config, configs[node.previous[1] - 1]); break;
            case LayerType::AVGPOOL2D: *inouts[node_id] = avgpool2d(input, config); break;
            case LayerType::BATCHNORM2D: *inouts[node_id] = batchnorm2d(input, get<Tensor<4,Scalar>>(node.layer.kernel()), config); break;
            case LayerType::CONV2D: *inouts[node_id] = conv2d(input, get<Tensor<6,Scalar>>(node.layer.kernel()), node.layer.bias(), config); break;
            case LayerType::LINEAR: *inouts[node_id] = linear(input, get<Tensor<4,Scalar>>(node.layer.kernel()), node.layer.bias(), config); break;
            case LayerType::POLY: *inouts[node_id] = poly(input, get<Tensor<4,Scalar>>(node.layer.kernel()), config); break;
            case LayerType::POLYSKIP: *inouts[node_id] = polyskip(input, *inouts[node.previous[1]], get<Tensor<4,Scalar>>(node.layer.kernel()), config, configs[node.previous[1] - 1]); break;
            default: throw invalid_argument("Model::compute: LayerType not implemented");
        }
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
    end = high_resolution_clock::now();
    auto time_span = double(duration_cast<milliseconds>(end - start).count()) / 1000;

    std::cout << "\t(in " << time_span << "s) level: " << (*inouts[node_id])[0][0].level() << " scale: " << (*inouts[node_id])[0][0].scale() << std::endl;
}

void Model::config(size_t node_id, unordered_set<size_t>& visited) // create configurations from layers
{
    size_t node_idx = node_id - 1;
    const Node& node = nodes[node_idx];

    // config unvisited dependencies
    for (size_t previous_id : node.previous)
        if (previous_id && visited.find(previous_id) == visited.end())
            config(previous_id, visited);
    
    std::cout << "Config node " << node_id << " " << to_string(node.layer.type()) << " Previous: " << node.previous << std::endl;
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

void Model::encode_biases()
{
    std::cout << "Encoding biases" << std::endl;
    biases.clear();
    biases.resize(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++)
    {
        std::cout << "Node " << (i+1) << " with shape " << nodes[i].layer.bias().shape() << " " << to_string(nodes[i].layer.type()) << std::endl;
        const auto& type = nodes[i].layer.type();
        switch (type)
        {
            case LayerType::ADD:
            case LayerType::AVGPOOL2D:
            case LayerType::BATCHNORM2D:
            case LayerType::POLY:
            case LayerType::POLYSKIP: break;
            case LayerType::CONV2D: break;
            case LayerType::LINEAR:
            {
                biases[i] = encode_bias(nodes[i].layer.bias(), configs[i], config::NTHREADS);
                break;
            }
            default: throw invalid_argument("Model::encode_kernels_and_biases: LayerType not implemented");
        }
    }
}

void Model::encode_kernels()
{
    std::cout << "Encoding kernels" << std::endl;
    int max_level;
    double keyscale;
    {
        auto pt = Plaintext(0);
        max_level = pt.level();
        keyscale = pt.keyscale();
    }
    kernels.clear();
    kernels.resize(nodes.size());
    kernel_maps.resize(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++)
    {
        int level = ((fhe::SUBLEVELS * max_level) - in_depths[i]) / fhe::SUBLEVELS;
        cout << "Node " << (i+1) << " with shape " << nodes[i].layer.kshape() << " " << to_string(nodes[i].layer.type()) << ", level = " << level << ", keyscale = " << keyscale << endl;
        const auto& type = nodes[i].layer.type();
        switch (type)
        {
            case LayerType::ADD:
            case LayerType::AVGPOOL2D: break;
            case LayerType::CONV2D: break;
            case LayerType::BATCHNORM2D:
            case LayerType::LINEAR:
            case LayerType::POLY:
            case LayerType::POLYSKIP: break;
            default: throw invalid_argument("Model::encode_kernels_and_biases: LayerType not implemented");
        }
    }
}

void Model::init()
{
    configs.resize(nodes.size());
    in_depths.resize(nodes.size());
    out_depths.resize(nodes.size());
    unordered_set<size_t> visited;
    config(last_id, visited);
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

Tensor<3,Scalar> Model::decode_output(const Tensor<2,Ciphertext>& output) const
{
    const auto& config = configs[last_id - 1];
    return fhe::hw::decode_output(output, config, config::NTHREADS);
}

Tensor<2,Ciphertext> Model::encode_input(const Tensor<3,Scalar>& input) const
{
    auto config = configs[first_id - 1];
    config.reset(); // move head to beginning and encode input
    return fhe::hw::encode_input(input, config, config::NTHREADS);
}

Encoding Model::encoding() const
{
    return Encoding::HW;
}

Tensor<2,Ciphertext> Model::forward(Tensor<2,Ciphertext>& input) const
{
    unordered_set<size_t> visited;
    unordered_map<size_t,shared_ptr<Tensor<2,Ciphertext>>> inouts;
    inouts[0] = make_shared<Tensor<2,Ciphertext>>(move(input));
    compute(last_id, visited, inouts);
    input = move(*inouts[0]);
    return *inouts[last_id];
}

Tensor<2,Ciphertext> Model::forward(const Tensor<2,Ciphertext>& input) const
{
    Tensor<2,Ciphertext> input_copy = input;
    return forward(input_copy);
}

void Model::print() const
{
    return BaseModel::print();
}

void Model::preencode_kernels()
{
    // This is only for CONV2D
    // convert kernels from (Co x Kh x Kw x Ci x Ih x Iw) to (Co x Kh x Kw x Ci x nCT x slots)
    std::cout << "Preecoding kernels" << std::endl;
    for (size_t i = 0; i < nodes.size(); i++)
    {
        auto& node = nodes[i];
        const auto& type = node.layer.type();
        if (type == LayerType::CONV2D)
        {
            std::cout << "Node " << (i+1) << " with shape " << nodes[i].layer.kshape() << " " << to_string(nodes[i].layer.type()) << std::endl;
            auto& w = get<Tensor<6,Scalar>>(node.layer.kernel());
            w = map_kernel(w, configs[i], config::NTHREADS);
        }
    }
}

} // hw

} // fhe
