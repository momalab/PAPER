#include "configuration.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>
#include "layer.h"
#include "math.h"
#include "plaintext.h"
#include "tensor.h"

using namespace ml;
using namespace std;
using namespace type;

namespace fhe
{

namespace hw
{

const int LINEAR_VERSION = 2; // use new mapping method

Configuration::Configuration(const vector<Layer>& layers) : Configuration(layers, Tensor<2,size_t>()) {}

Configuration::Configuration(const vector<Layer>& layers, Tensor<2,size_t> initial_mapping) : Configuration(layers, initial_mapping, 0UL, vector<size_t>()) {}

Configuration::Configuration(const vector<Layer>& layers, const vector<size_t>& padding) : Configuration(layers, Tensor<2,size_t>(), 0UL, padding) {}

Configuration::Configuration(const vector<Layer>& layers, Tensor<2,size_t> initial_mapping, size_t initial_nCTs) : Configuration(layers, initial_mapping, initial_nCTs, vector<size_t>()) {}

Configuration::Configuration(const vector<Layer>& layers, Tensor<2,size_t> initial_mapping, size_t initial_nCTs, const vector<size_t>& padding)
{
    if (layers.empty()) throw invalid_argument("fhe::hw::Configuration: layers must not be empty");
    this->layers.resize(layers.size() + 1);
    set_input_layer(layers[0].ishape());
    copy(layers.begin(), layers.end(), this->layers.begin() + 1);

    if (initial_mapping.empty())
    {
        const auto& ishape = layers[0].ishape();
        const auto& ih = ishape[1];
        const auto& iw = ishape[2];
        initial_mapping = Tensor<2,size_t>{ih, iw};
        if (padding.empty()) for (size_t i = 0; i < ih; i++) iota(initial_mapping[i].begin(), initial_mapping[i].end(), i * iw);
        else
        {
            if (padding.size() != 2) throw invalid_argument("fhe::hw::Configuration: padding must be of size 2");
            const auto& pw = padding[1];
            for (size_t i = 0; i < ih; i++)
                for (size_t j = 0; j < iw; j++)
                    initial_mapping[i][j] = i * (iw + pw) + j;
        }
    }
    init_mapping(initial_mapping);

    init_nCT(initial_nCTs);
    this->index = 0;
}

Scalar Configuration::divisor() const
{
    return layers[index].divisor();
}

size_t Configuration::get() const
{
    return index;
}

void Configuration::init_mapping(const Tensor<2,size_t>& initial_mapping)
{
    size_t isize = layers.size() + 1;
    mappings.resize(isize);
    mappings[0] = initial_mapping;
    for (size_t i = 1; i < isize; i++) remap(i);
}

void Configuration::init_nCT(size_t initial_nCTs)
{
    const auto& slots = Plaintext::default_slots();
    size_t isize = layers.size() + 1;
    nCTs.resize(isize);
    if (initial_nCTs) nCTs[0] = initial_nCTs;
    else
    {
        size_t nslots = mappings[0].back().back() + 1;
        nCTs[0] = nslots / slots + (nslots % slots != 0);
    }
    for (size_t i = 1; i < isize; i++)
    {
        if (layers[i-1].type() == LayerType::LINEAR)
        {
            switch (LINEAR_VERSION)
            {
                case 0:
                {
                    // Previous method
                    const auto& ko = layers[i-1].kshape()[0];
                    size_t max_slot = min(1UL << util::ceil_log2(mappings[i-1].back().back() + 1), Plaintext::default_slots());
                    size_t nslots = ko * max_slot;
                    nCTs[i] = nslots / slots + (nslots % slots != 0);
                    break;
                }
                case 1:
                {
                    // New method
                    size_t nslots = mappings[i].back().back() + 1;
                    nCTs[i] = nslots / slots + (nslots % slots != 0);
                    break;
                }
                case 2:
                {
                    nCTs[i] = 1; // One result per output channel
                    break;
                }
                default: throw invalid_argument("Configuration: LINEAR_VERSION not valid");
            }
        }
        else nCTs[i] = nCTs[i-1]; // this can be optimized to drop the last ciphertexts according to max(mapping)
    }
}

const Tensor<2,size_t>& Configuration::imap() const
{
    return mappings[index];
}

const vector<size_t>& Configuration::ishape() const
{
    return layers[index].ishape();
}

const vector<size_t>& Configuration::kshape() const
{
    return layers[index].kshape();
}

const Tensor<2,size_t>& Configuration::mapping() const
{
    return imap();
}

vector<size_t> Configuration::max_padding() const
{
    size_t max_ph = 0, max_pw = 0;
    for (const auto& layer : layers)
    {
        const auto& padding = layer.padding();
        max_ph = max(max_ph, padding[0]);
        max_pw = max(max_pw, padding[1]);
    }
    return {max_ph, max_pw};
}

const size_t& Configuration::nct() const
{
    return ncto();
}

const size_t& Configuration::ncti() const
{
    return nCTs[index];
}

const size_t& Configuration::ncto() const
{
    return nCTs[index+1];
}

void Configuration::next()
{
    index++;
}

const Tensor<2,size_t>& Configuration::omap() const
{
    return mappings[index+1];
}

const vector<size_t>& Configuration::oshape() const
{
    return layers[index].oshape();
}

const vector<size_t>& Configuration::padding() const
{
    return layers[index].padding();
}

void Configuration::pop_back()
{
    if (layers.size() == 1) throw invalid_argument("Configuration: no layer to pop");
    layers.pop_back();
    init_mapping(mappings[0]);
    init_nCT();
}

void Configuration::previous()
{
    if (index == 0) throw invalid_argument("Configuration: index is 0");
    index--;
}

void Configuration::push_back(const Layer& layer)
{
    layers.push_back(layer);
    init_mapping(mappings[0]);
    init_nCT();
}

void Configuration::remap(size_t index)
{
    if (index == 0) throw invalid_argument("Configuration: index is 0");

    const auto& layer = layers[index-1];
    const auto& iw = layer.ishape()[2];
    const auto& sh = layer.stride()[0];
    const auto& sw = layer.stride()[1];
    const auto& oh = layer.oshape()[1];
    const auto& ow = layer.oshape()[2];

    mappings[index] = Tensor<2,size_t>{oh, ow};
    switch (layers[index-1].type())
    {
        case LayerType::AVGPOOL2D:
        case LayerType::CONV2D:
        {
            for (size_t ii = 0, bi = 0; bi < oh; ii += sh, bi++)
            {
                for (size_t ij = 0, bj = 0; bj < ow; ij += sw, bj++)
                {
                    size_t idx = ii * iw + ij;
                    mappings[index][bi][bj] = mappings[index-1][idx / iw][idx % iw];
                }
            }
            break;
        }
        case LayerType::LINEAR:
        {
            switch (LINEAR_VERSION)
            {
                case 0:
                {
                    // Previous method
                    size_t max_slot = min(1UL << util::ceil_log2(mappings[index-1].back().back() + 1), Plaintext::default_slots());
                    for (size_t o = 0; o < ow; o++) mappings[index][0][o] = o * max_slot;
                    break;
                }
                case 1:
                {
                    // New method
                    auto& previous_mapping = mappings[index-1];
                    const auto& ih = previous_mapping.size();
                    const auto& iw = previous_mapping[0].size();
                    size_t max_mapsize = ow * (previous_mapping.back().back() + 1);
                    vector<bool> free_map(max_mapsize, true);
                    int offset = -1;
                    for (size_t o = 0; o < ow; o++)
                    {
                        // find offset
                        for (bool is_free = false; !is_free;)
                        {
                            offset++;
                            is_free = true;
                            for (size_t ii = 0; ii < ih && is_free; ii++)
                                for (size_t ij = 0; ij < iw && is_free; ij++)
                                    is_free = free_map[previous_mapping[ii][ij] + offset];
                        }
                        // set free_map
                        for (auto& row : previous_mapping)
                            for (auto& idx : row)
                                free_map[idx + offset] = false;
                        // set mapping
                        mappings[index][0][o] = offset;
                    }
                    break;
                }
                case 2:
                {
                    // New new method
                    mappings[index][0][0] = 0;
                    break;
                }
                default: throw invalid_argument("Configuration: LINEAR_VERSION not valid");
            }
            break;
        }
        case LayerType::ADD:
        case LayerType::BATCHNORM2D:
        case LayerType::POLY:
        case LayerType::POLYSKIP:
        {
            mappings[index] = mappings[index-1];
            break;
        }
        default: throw invalid_argument("Configuration: LayerType not implemented");
    }
}

void Configuration::reset()
{
    index = 0;
}

void Configuration::set(size_t index)
{
    if (index >= layers.size()) throw invalid_argument("Configuration: index out of range");
    this->index = index;
}

void Configuration::set_input_layer(const vector<size_t>& ishape)
{
    if (this->layers.empty()) this->layers.resize(1);
    LayerType type = LayerType::AVGPOOL2D;
    Tensor<4,Scalar> kernel;
    Tensor<3,Scalar> bias;
    const auto& ci = ishape[0];
    vector<size_t> kshape{ci, ci, 1, 1};
    vector<size_t> bshape;
    vector<size_t> stride{1, 1};
    vector<size_t> padding{0, 0};
    Scalar divisor = 1.0;
    this->layers[0] = Layer(type, kernel, bias, ishape, kshape, bshape, stride, padding, divisor);
}

int Configuration::shift(size_t krow, size_t kcol) const
{
    const auto& ish = ishape();
    const auto& pad = padding();
    const auto& map = mapping();
    const auto ih = ish[1];
    const auto iw = ish[2];
    const auto ph = pad[0];
    const auto pw = pad[1];
    const int offset = map[ph][pw];

    int correction = 0;
    if (krow >= ih)
    {
        int row_distance = int(map[1][0] - map[0][0]);
        correction += (krow - ih + 1) * row_distance; // shift correct for input 
        krow = ih - 1; // set kernel row to last valid input row
    }
    if (kcol >= iw)
    {
        int col_distance = int(map[0][1] - map[0][0]);
        correction += (kcol - iw + 1) * col_distance; // shift correct for input 
        kcol = iw - 1; // set kernel column to last valid input column
    }
    
    return int(map[krow][kcol]) + correction - offset;
}

const vector<size_t>& Configuration::stride() const
{
    return layers[index].stride();
}

const LayerType& Configuration::type() const
{
    return layers[index].type();
}

} // hw

} // fhe
