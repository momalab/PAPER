#include "model_base.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include "layer.h"
#include "tensor.h"

using namespace fhe;
using namespace std;
using namespace type;

namespace ml
{

inline Scalar invert(const Scalar& divisor)
{
    if (divisor == 0.0) throw runtime_error("BaseModel::Cannot invert zero");
    return 1.0 / divisor; // invert scalar
}

inline Tensor<3,Scalar> invert(const Tensor<3,Scalar>& coeff)
{
    Tensor<3,Scalar> r = coeff;
    for (auto& m : r) for (auto& v : m) for (auto& e : v) e = 1.0 / e; // invert coeff
    return r;
}

inline Tensor<3,Scalar> sqrt(const Tensor<3,Scalar>& coeff)
{
    Tensor<3,Scalar> r = coeff;
    for (auto& m : r) for (auto& v : m) for (auto& e : v) e = std::sqrt(e); // sqrt coeff
    return r;
}

// Constructors

BaseModel::BaseModel(const nlohmann::json& json, const vector<size_t>& ishape)
{
    init(json, ishape);
}

BaseModel::BaseModel(ifstream is, const vector<size_t>& ishape)
{
    nlohmann::json json;
    is >> json;
    init(json, ishape);
}

BaseModel::BaseModel(const string& filename, const vector<size_t>& ishape) : BaseModel(ifstream(filename), ishape) {}

void BaseModel::fuse_nodes()
{
    bool conv_batch_priority = true;
    if (conv_batch_priority) fuse_conv_batch();
    fuse_poly_add_batch();
    fuse_poly_batch();
    if (!conv_batch_priority) fuse_conv_batch();
    remove_isolated_nodes();
}

Encoding BaseModel::encoding() const
{
    throw std::runtime_error("BaseModel::encoding() not implemented");
}

void BaseModel::fuse_conv_batch()
{
    // search for conv2d followed by batchnorm2d
    for (size_t node_id = 1; node_id <= nodes.size(); node_id++)
    {
        Node& node = nodes[node_id - 1];
        if (node.layer.type() == LayerType::CONV2D && node.next.size() == 1)
        {
            size_t next_id = node.next[0];
            Node& next_node = nodes[next_id - 1];
            if (next_node.layer.type() == LayerType::BATCHNORM2D)
            {
                // fuse
                node.layer = Layer::fuse(node.layer, next_node.layer);
                node.next = next_node.next;
                // change previous of the next nodes of the next node from next_id to node_id
                for (size_t next_next_id : node.next)
                {
                    Node& next_next_node = nodes[next_next_id - 1];
                    for (size_t i = 0; i < next_next_node.previous.size(); i++)
                        if (next_next_node.previous[i] == next_id) next_next_node.previous[i] = node_id;
                }
                // clear previous and next of the next node
                next_node.previous.clear();
                next_node.next.clear();
            }
        }
    }
}

void BaseModel::fuse_poly_add_batch()
{
    // search for add layers followed by a single poly and preceded by two batchnorm2d
    for (size_t node_id = 1; node_id <= nodes.size(); node_id++)
    {
        Node& node = nodes[node_id - 1];
        if (node.layer.type() == LayerType::ADD && node.previous.size() == 2 && node.next.size() == 1)
        {
            // previous nodes
            size_t prev_id1 = node.previous[0];
            size_t prev_id2 = node.previous[1];
            if (!prev_id1 || !prev_id2) continue; // input node
            Node& prev_node1 = nodes[prev_id1 - 1];
            Node& prev_node2 = nodes[prev_id2 - 1];

            bool is_batch1 = prev_node1.layer.type() == LayerType::BATCHNORM2D;
            bool is_batch2 = prev_node2.layer.type() == LayerType::BATCHNORM2D;
            if (is_batch2 && !is_batch1)
            {
                swap(node.previous[0], node.previous[1]);
                node_id--;
                continue;
            }
            // next node
            size_t next_id = node.next[0];
            Node& next_node = nodes[next_id - 1];
            
            if (next_node.layer.type() == LayerType::POLY && is_batch1)
            {
                // fuse
                if (is_batch2)
                {
                    node.layer = Layer::fuse(prev_node1.layer, prev_node2.layer, node.layer, next_node.layer);
                    node.previous = {prev_node1.previous[0], prev_node2.previous[0]};
                }
                else
                {
                    node.layer = Layer::fuse(prev_node1.layer, node.layer, next_node.layer);
                    node.previous = {prev_node1.previous[0], node.previous[1]};
                }
                node.next = next_node.next;
                // change previous of the next nodes of the next node from next_id to node_id
                for (size_t next_next_id : node.next)
                {
                    Node& next_next_node = nodes[next_next_id - 1];
                    for (size_t i = 0; i < next_next_node.previous.size(); i++)
                        if (next_next_node.previous[i] == next_id) next_next_node.previous[i] = node_id;
                }
                // change the next of the previous nodes of the previous nodes from prev_id1 and prev_id2 to node_id
                auto update_previous_nodes = [&](const Node& prev_node, size_t prev_id)
                {
                    for (size_t prev_prev_id : prev_node.previous)
                    {
                        Node& prev_prev_node = nodes[prev_prev_id - 1];
                        for (size_t i = 0; i < prev_prev_node.next.size(); i++)
                            if (prev_prev_node.next[i] == prev_id) prev_prev_node.next[i] = node_id;
                    }
                };
                update_previous_nodes(prev_node1, prev_id1);
                if (is_batch2)
                {
                    update_previous_nodes(prev_node2, prev_id2);
                    prev_node2.previous.clear();
                    prev_node2.next.clear();
                }
                // clear previous and next of the previous nodes and the next node
                prev_node1.previous.clear();
                prev_node1.next.clear();
                next_node.previous.clear();
                next_node.next.clear();
            }
        }
    }
}

void BaseModel::fuse_poly_batch()
{
    // search for batchnorm2d
    // fuse if it has a single next node of type poly
    for (size_t node_id = 1; node_id <= nodes.size(); node_id++)
    {
        Node& node = nodes[node_id - 1];
        if (node.layer.type() == LayerType::BATCHNORM2D && node.next.size() == 1)
        {
            size_t next_id = node.next[0];
            Node& next_node = nodes[next_id - 1];
            if (next_node.layer.type() == LayerType::POLY)
            {
                // fuse
                node.layer = Layer::fuse(node.layer, next_node.layer);
                node.next = next_node.next;
                // change previous of the next nodes of the next node from next_id to node_id
                for (size_t next_next_id : node.next)
                {
                    Node& next_next_node = nodes[next_next_id - 1];
                    for (size_t i = 0; i < next_next_node.previous.size(); i++)
                        if (next_next_node.previous[i] == next_id) next_next_node.previous[i] = node_id;
                }
                // clear next_node.previous and next_node.next
                next_node.previous.clear();
                next_node.next.clear();
            }
        }
    }
}

const vector<BaseModel::Node>& BaseModel::get_nodes() const
{
    return nodes;
}

size_t BaseModel::get_last_id() const
{
    return last_id;
}

void BaseModel::handle_avgpool2d(size_t node_id)
{
    Node& node = nodes[node_id - 1];
    if (node.layer.divisor() == 1.0) return; // no need to change weights

    bool solvable = !node.next.empty(); // if there are no next nodes, the model is not solvable forward
    for (size_t next_id : node.next) if (!(solvable = is_solvable_forward(next_id))) break;
    if (solvable) for (size_t next_id : node.next) update_kernel_forward(next_id, 1.0 / node.layer.divisor());
    else
    {
        for (size_t prev_id : node.previous) if (!(solvable = is_solvable_backward(prev_id, node_id))) break;
        if (solvable) for (size_t prev_id : node.previous) update_kernel_backward(prev_id, node_id, 1.0 / node.layer.divisor());
    }

    if (solvable) node.layer.divisor() = 1.0;
}

void BaseModel::init(const nlohmann::json& json, const vector<size_t>& ishape)
{
    for (const auto& layer_desc : json) // build layers
    {
        this->nodes.emplace_back(Node());
        Node& node = nodes.back();
        node.previous = layer_desc["previous"].get<vector<size_t>>();
        node.next = layer_desc["next"].get<vector<size_t>>();
        node.layer = Layer(layer_desc, node.previous[0] ? nodes[node.previous[0] - 1].layer.oshape() : ishape);
        if (node.next.empty()) last_id = nodes.size();
    }

    if (!last_id) throw invalid_argument("BaseModel::init: Output layer not found");

    fuse_nodes();
    unordered_set<size_t> visited;
    redistribute_weights(last_id, visited);
}

bool BaseModel::is_solvable_backward(size_t node_id, size_t origin_id) const
{
    if (!node_id) return false; // input node
    const Node& node = nodes[node_id - 1];

    if (node.layer.has_updatable_kernel())
    {
        if (node.next.size() > 1) { for (size_t next_id : node.next) if (!is_solvable_bypass(next_id, origin_id)) return false; }
        else return true;
    }
    else if (node.next.size() > 1) return false;
    else for (size_t previous_id : node.previous) if (!is_solvable_backward(previous_id, origin_id)) return false;

    return true;
}

bool BaseModel::is_solvable_bypass(size_t node_id, size_t origin_id) const
{
    const Node& node = nodes[node_id - 1];
    if (node.previous.size() > 1) return false; // avoid complicated path finding
    if (node_id == origin_id || node.layer.has_updatable_kernel()) return true;
    for (size_t next_id : node.next) if (!is_solvable_bypass(next_id, origin_id)) return false;
    return true;
}

bool BaseModel::is_solvable_forward(size_t node_id) const
{
    const Node& node = nodes[node_id - 1];

    // check if this node has more than one previous node
    if (node.previous.size() > 1 && node.layer.type() != LayerType::POLYSKIP) return false;

    // check if this node has weights
    if (node.layer.has_updatable_kernel()) return true;

    // otherwise, check if next nodes are solvable
    for (size_t next_id : node.next) if (!is_solvable_forward(next_id)) return false;

    return true;
}

void BaseModel::print() const
{
    for (size_t i = 0; i < nodes.size(); i++)
    {
        const Node& node = nodes[i];
        const Layer& layer = node.layer;
        cout << "Node " << i + 1 << " Previous: " << node.previous << " Next: " << node.next << " " << to_string(layer.type());
        switch (node.layer.type())
        {
            case LayerType::ADD: break;
            case LayerType::AVGPOOL2D: cout << " kshape: " << layer.kshape() << " padding: " << layer.padding() << " stride: " << layer.stride(); break;
            case LayerType::CONV2D: cout << " kshape: " << layer.kshape() << " bshape: " << layer.bshape() << " padding: " << layer.padding() << " stride: " << layer.stride(); break;
            case LayerType::LINEAR: cout << " kshape: " << layer.kshape() << " bshape: " << layer.bshape(); break;
            case LayerType::BATCHNORM2D:
            case LayerType::POLY:
            case LayerType::POLYSKIP: cout << " ksize: " << get<Tensor<4,Scalar>>(layer.kernel()).size(); break;
            default: throw invalid_argument("BaseModel::print: LayerType not implemented");
        }
        cout << endl;
    }
}

void BaseModel::redistribute_weights(size_t node_id, unordered_set<size_t>& visited)
{
    if (!node_id || visited.find(node_id) != visited.end()) return;
    
    Node& node = nodes[node_id - 1];

    // solve previous nodes
    for (size_t previous_id : node.previous)
        if (previous_id && visited.find(previous_id) == visited.end())
            redistribute_weights(previous_id, visited);

    // solve this node
    solve(node_id);

    // mark this node as visited
    visited.insert(node_id);
}

void BaseModel::remove_isolated_nodes()
{
    for (size_t node_id = 1; node_id <= nodes.size(); node_id++)
    {
        size_t i = node_id - 1;
        Node& node = nodes[i];
        if (node.previous.empty() && node.next.empty())
        {
            if (node_id != last_id) last_id--;

            // update previous and next nodes (decrease if > node_id)
            for (size_t node_jd = 1; node_jd <= nodes.size(); node_jd++)
            {
                size_t j = node_jd - 1;
                Node& node_j = nodes[j];
                for (auto& previous_id : node_j.previous) if (previous_id > node_id) previous_id--;
                for (auto& next_id : node_j.next) if (next_id > node_id) next_id--;
            }

            nodes.erase(nodes.begin() + i);
            i--;
        }
    }
}

void BaseModel::solve(size_t node_id)
{
    Node& node = nodes[node_id - 1];
    
    const auto& layer_type = node.layer.type();
    if (layer_type == LayerType::AVGPOOL2D) return handle_avgpool2d(node_id);
    if (layer_type != LayerType::BATCHNORM2D && layer_type != LayerType::POLY &&
        layer_type != LayerType::POLYSKIP) return; // no weights to change

    auto& kernel = get<Tensor<4,Scalar>>(node.layer.kernel());
    const auto& coeff = kernel.back();

    bool solvable = !node.next.empty(); // if there are no next nodes, the model is not solvable forward
    for (size_t next_id : node.next) if (!(solvable = is_solvable_forward(next_id))) break;
    if (solvable) // solvable forward
    {
        for (size_t next_id : node.next) update_kernel_forward(next_id, coeff);
        auto coeff_divisor = invert(coeff);
        kernel.resize(kernel.size() - 1); // remove the largest order coefficient since it is 1
        for (size_t i = 0; i < kernel.size(); i++) kernel[i] *= coeff_divisor;
        return;
    }

    for (size_t prev_id : node.previous) if (!(solvable = is_solvable_backward(prev_id, node_id))) break;
    if (solvable) // solvable backward
    {
        auto adjusted_coeff = (layer_type == LayerType::BATCHNORM2D) ? coeff : sqrt(coeff);
        for (size_t prev_id : node.previous) update_kernel_backward(prev_id, node_id, adjusted_coeff);
        auto coeff_divisor = invert(adjusted_coeff);
        kernel.resize(kernel.size() - 1); // remove the largest order coefficient since it is 1
        switch (layer_type)
        {
            case LayerType::BATCHNORM2D: break; // b1 is dropped and b0 does not change
            case LayerType::POLY: // c2 is dropped, c1 is updated, and c0 does not change
            {
                kernel[1] *= coeff_divisor; // update c1
                break;
            }
            case LayerType::POLYSKIP: // k5 is dropped, {k3, k2} are updated, and {k4, k1, k0} do not change
            {
                kernel[2] *= coeff_divisor; // update k2
                kernel[3] *= coeff_divisor; // update k3
                break;
            }
            default: throw invalid_argument("BaseModel::solve: LayerType not suppoted");
        }
        return;
    }

    // special case for POLY
    // C -> P -> C -> ... C -> A -> P
    //        -> C             ^
    if (layer_type == LayerType::POLY && node.previous.size() == 1)
    {
        // check pattern

        // check if previous node is ADD
        size_t add_id = node.previous[0];
        const Node& add_node = nodes[add_id - 1];
        if (add_node.layer.type() != LayerType::ADD) return; // not a special case

        // check if ADD's previous nodes are CONV2D and POLY
        size_t conv_id = add_node.previous[0];
        size_t poly_id = add_node.previous[1];
        if (nodes[conv_id - 1].layer.type() != LayerType::CONV2D) swap(conv_id, poly_id);
        Node* conv_node = &nodes[conv_id - 1];
        Node* poly_node = &nodes[poly_id - 1];
        if (conv_node->layer.type() != LayerType::CONV2D || poly_node->layer.type() != LayerType::POLY) return; // not a special case

        // check if POLY node has two next nodes and one previous node
        if (poly_node->next.size() != 2 || poly_node->previous.size() != 1) return; // not a special case

        // check if one of the next nodes of POLY is a CONV2D
        Node* poly_conv_node = &nodes[poly_node->next[0] - 1];
        if (poly_conv_node->layer.type() != LayerType::CONV2D) poly_conv_node = &nodes[poly_node->next[1] - 1];
        if (poly_conv_node->layer.type() != LayerType::CONV2D) return; // not a special case

        // update nodes
        auto adjusted_coeff = sqrt(coeff);
        auto coeff_divisor = invert(adjusted_coeff);
        
        // update conv node before add
        conv_node->layer.update_kernel_backward(adjusted_coeff);
        
        // update conv node after poly
        poly_conv_node->layer.update_kernel_forward(coeff_divisor);

        // update poly node
        auto& poly_kernel = get<Tensor<4,Scalar>>(poly_node->layer.kernel());
        poly_kernel[0] *= adjusted_coeff; // update c0
        poly_kernel[1] *= adjusted_coeff; // update c1
        if (poly_kernel.size() > 2) poly_kernel[2] *= adjusted_coeff; // update c2 if exists
        else poly_kernel.emplace_back(adjusted_coeff); // add c2 if not exists
        
        // update this node
        kernel.resize(kernel.size() - 1); // remove the largest order coefficient since it is 1
        kernel[1] *= coeff_divisor; // update c1

        // solve POLY node
        solve(poly_id);
    }
}

void BaseModel::update_kernel_backward(size_t node_id, size_t origin_id, const Scalar& scale)
{
    Node& node = nodes[node_id - 1];
    if (node.layer.has_updatable_kernel())
    {
        node.layer.update_kernel_backward(scale);
        const Scalar scale_bypass = 1.0 / scale;
        for (size_t next_id : node.next) update_kernel_bypass(next_id, origin_id, scale_bypass);
    }
    else for (size_t previous_id : node.previous) update_kernel_backward(previous_id, origin_id, scale);
}

void BaseModel::update_kernel_backward(size_t node_id, size_t origin_id, const Tensor<3,Scalar>& scale)
{
    Node& node = nodes[node_id - 1];
    if (node.layer.has_updatable_kernel())
    {
        node.layer.update_kernel_backward(scale);
        auto scale_bypass = invert(scale);
        for (size_t next_id : node.next) update_kernel_bypass(next_id, origin_id, scale_bypass);
    }
    else for (size_t previous_id : node.previous) update_kernel_backward(previous_id, origin_id, scale);
}

void BaseModel::update_kernel_bypass(size_t node_id, size_t origin_id, const Scalar& scale)
{
    if (node_id == origin_id) return;
    Node& node = nodes[node_id - 1];
    if (node.layer.has_updatable_kernel()) node.layer.update_kernel_forward(scale);
    else for (size_t next_id : node.next) update_kernel_bypass(next_id, origin_id, scale);
}

void BaseModel::update_kernel_bypass(size_t node_id, size_t origin_id, const Tensor<3,Scalar>& scale)
{
    if (node_id == origin_id) return;
    Node& node = nodes[node_id - 1];
    if (node.layer.has_updatable_kernel()) node.layer.update_kernel_forward(scale);
    else for (size_t next_id : node.next) update_kernel_bypass(next_id, origin_id, scale);
}

void BaseModel::update_kernel_forward(size_t node_id, const Scalar& scale)
{
    Node& node = nodes[node_id - 1];
    if (node.layer.has_updatable_kernel()) node.layer.update_kernel_forward(scale);
    else for (size_t next_id : node.next) update_kernel_forward(next_id, scale);
}

void BaseModel::update_kernel_forward(size_t node_id, const Tensor<3,Scalar>& scale)
{
    Node& node = nodes[node_id - 1];
    if (node.layer.has_updatable_kernel()) node.layer.update_kernel_forward(scale);
    else for (size_t next_id : node.next) update_kernel_forward(next_id, scale);
}

} // ml