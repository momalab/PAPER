#include "layer.h"

#include <cstddef>
#include <nlohmann/json.hpp>
#include <string>
#include <variant>
#include <vector>
#include "tensor.h"
#include "scalar.h"

using namespace fhe;
using namespace std;
using namespace type;
using json = nlohmann::json;

namespace ml
{

inline bool is_coeff_uniform(const Tensor<3,Scalar>& b)
{
    const auto shape = b.shape();
    const auto& ci = shape[0];
    const auto& ih = shape[1];
    const auto& iw = shape[2];
    for (size_t i = 0; i < ci; i++)
        for (size_t ii = 0; ii < ih; ii++)
            for (size_t ij = 0; ij < iw; ij++)
                if (b[i][0][0] != b[i][ii][ij]) return false;
    return true;
}

inline bool is_coeff_uniform(const Tensor<4,Scalar>& b, int start_coeff = 0)
{
    const auto ncoeffs = b.shape()[0];
    for (size_t c = start_coeff; c < ncoeffs; c++)
        if (!is_coeff_uniform(b[c])) return false;
    return true;
}

// Constructors

Layer::Layer(const nlohmann::json& desc, const vector<size_t>& ishape)
{
    _type = to_layer(desc["layer"]);
    _ishape = ishape;
    _divisor = 1.0;
    if (!_kernel) _kernel = make_shared<variant<Tensor<4,Scalar>,Tensor<6,Scalar>>>();
    if (!_bias) _bias = make_shared<Tensor<3,Scalar>>();
    switch (_type)
    {
        case LayerType::ADD: init_add(); break;
        case LayerType::AVGPOOL2D: init_avgpool2d(desc); break;
        case LayerType::BATCHNORM2D: init_batchnorm(desc); break;
        case LayerType::CONV2D: init_conv2d(desc); break;
        case LayerType::LINEAR: init_linear(desc); break;
        case LayerType::POLY: init_poly(desc); break;
        default: throw invalid_argument("ml::Layer::Layer: Invalid LayerType");
    }
}

Layer::Layer
(
    const LayerType& type, const variant<Tensor<4,Scalar>,Tensor<6,Scalar>>& kernel, const Tensor<3,Scalar>& bias,
    const vector<size_t>& ishape, const vector<size_t>& kshape, const vector<size_t>& bshape,
    const vector<size_t>& stride, const vector<size_t>& padding, const Scalar& divisor
)
{
    _type = type;
    _ishape = ishape;
    _kshape = kshape;
    _bshape = bshape;
    _stride = stride;
    _padding = padding;
    _divisor = divisor;
    if (!_kernel) _kernel = make_shared<variant<Tensor<4,Scalar>,Tensor<6,Scalar>>>();
    *_kernel = kernel;
    if (!_bias) _bias = make_shared<Tensor<3,Scalar>>();
    *_bias = bias;

    switch (_type)
    {
        case LayerType::ADD: 
        case LayerType::BATCHNORM2D:
        case LayerType::POLY:
        case LayerType::POLYSKIP: _oshape = _ishape; break;
        case LayerType::CONV2D:
        case LayerType::LINEAR: _oshape = _bshape; break;
        case LayerType::AVGPOOL2D:
        {
            const auto& ci = _ishape[0];
            const auto& ih = _ishape[1];
            const auto& iw = _ishape[2];
            const auto& kh = _kshape[2];
            const auto& kw = _kshape[3];
            const auto& sh = _stride[0];
            const auto& sw = _stride[1];
            const auto& ph = _padding[0];
            const auto& pw = _padding[1];
            const auto& oh = (ih + 2UL*ph - kh) / sh + 1UL;
            const auto& ow = (iw + 2UL*pw - kw) / sw + 1UL;
            _oshape = {ci, oh, ow};
            break;
        }
        default: throw invalid_argument("ml::Layer::Layer: Invalid LayerType");
    }
}

// Private functions

void Layer::init_add()
{
    _oshape = _ishape;
    const auto& ci = _ishape[0];
    _kshape = {ci, ci, 1, 1};
}

void Layer::init_avgpool2d(const json& desc)
{
    const auto& kdata = desc["kernel"].get<vector<size_t>>();
    _stride = desc["stride"].get<vector<size_t>>();
    _padding = desc["padding"].get<vector<size_t>>();
    _divisor = desc["divisor"].get<Scalar>();
    const auto& ci = _ishape[0];
    const auto& ih = _ishape[1];
    const auto& iw = _ishape[2];
    const auto& kh = kdata[0];
    const auto& kw = kdata[1];
    const auto& sh = _stride[0];
    const auto& sw = _stride[1];
    const auto& ph = _padding[0];
    const auto& pw = _padding[1];
    const auto& oh = (ih + 2UL*ph - kh) / sh + 1UL;
    const auto& ow = (iw + 2UL*pw - kw) / sw + 1UL;
    _kshape = {ci, ci, kh, kw};
    _oshape = {ci, oh, ow};
}

void Layer::init_batchnorm(const json& desc)
{
    const auto& mu = desc["mu"].get<vector<Scalar>>();
    const auto& var = desc["var"].get<vector<Scalar>>();
    const auto& gamma = desc["gamma"].get<vector<Scalar>>();
    const auto& beta = desc["beta"].get<vector<Scalar>>();
    const auto& epsilon = desc["epsilon"].get<Scalar>();
    const auto& ci = _ishape[0];
    const auto& ih = _ishape[1];
    const auto& iw = _ishape[2];
    *_kernel = Tensor<4,Scalar>{2UL, ci, ih, iw};
    auto& w = get<Tensor<4,Scalar>>(*_kernel);
    for (size_t i = 0; i < ci; i++)
    {
        for (size_t ii = 0; ii < ih; ii++)
        {
            for (size_t ij = 0; ij < iw; ij++)
            {
                w[1][i][ii][ij] = gamma[i] / sqrt(var[i] + epsilon);
                w[0][i][ii][ij] = beta[i] - mu[i] * w[1][i][ii][ij];
            }
        }
    }
    _kshape = w.shape();
    _stride = {1, 1};
    _padding = {0, 0};
    _oshape = _ishape;
}

void Layer::init_conv2d(const json& desc)
{
    auto w = Tensor<4,Scalar>::copy(desc["kernel"].get<vector<vector<vector<vector<Scalar>>>>>());
    _kshape = w.shape();
    _stride = desc["stride"].get<vector<size_t>>();
    _padding = desc["padding"].get<vector<size_t>>();
    const auto& ci = _ishape[0];
    const auto& ih = _ishape[1];
    const auto& iw = _ishape[2];
    const auto& co = _kshape[0];
    const auto& kh = _kshape[2];
    const auto& kw = _kshape[3];
    const auto& sh = _stride[0];
    const auto& sw = _stride[1];
    const auto& ph = _padding[0];
    const auto& pw = _padding[1];
    const auto& oh = (ih + 2UL*ph - kh) / sh + 1UL;
    const auto& ow = (iw + 2UL*pw - kw) / sw + 1UL;
    *_kernel = Tensor<6,Scalar>{co, kh, kw, ci, ih, iw};
    auto& k = get<Tensor<6,Scalar>>(*_kernel);
    for (size_t o = 0; o < co; o++)
        for (size_t krow = 0; krow < kh; krow++)
            for (size_t kcol = 0; kcol < kw; kcol++)
                for (size_t i = 0; i < ci; i++)
                    for (size_t ii = 0; ii < ih; ii++)
                        for (size_t ij = 0; ij < iw; ij++)
                            k[o][krow][kcol][i][ii][ij] = w[o][i][krow][kcol];

    const auto& bdata = desc["bias"];
    if (co != bdata.size() && bdata.size()) throw invalid_argument("ml::Layer::init_conv2d: Invalid bias size (" + to_string(co) + " != " + to_string(bdata.size()) + ")");
    auto& b = *_bias;
    b = Tensor<3,Scalar>{co, oh, ow};
    if (bdata.size())
        for (size_t o = 0; o < co; o++)
            for (size_t i = 0; i < oh; i++)
                for (size_t j = 0; j < ow; j++)
                    b[o][i][j] = bdata[o];
    _bshape = b.shape();
    _oshape = _bshape;
}

void Layer::init_linear(const json& desc)
{
    const auto& kdata = desc["kernel"].get<vector<vector<Scalar>>>();
    const auto& bdata = desc["bias"].get<vector<Scalar>>();
    const auto& ci = _ishape[0];
    const auto& ih = _ishape[1];
    const auto& iw = _ishape[2];
    const auto& ko = kdata.size();
    if (ko != bdata.size()) throw invalid_argument("ml::Layer::init_linear: Invalid kernel/bias size");

    *_kernel = Tensor<4,Scalar>{ko, ci, ih, iw};
    auto& w = get<Tensor<4,Scalar>>(*_kernel);
    for (size_t o = 0; o < ko; o++)
        for (size_t i = 0; i < ci; i++)
            for (size_t ii = 0; ii < ih; ii++)
                for (size_t ij = 0; ij < iw; ij++)
                    w[o][i][ii][ij] = kdata[o][i*ih*iw + ii*iw + ij];
    _kshape = w.shape();

    auto& b = *_bias;
    b = Tensor<3,Scalar>{ko, 1UL, 1UL};
    for (size_t ii = 0; ii < ko; ii++)
        b[ii][0][0] = bdata[ii];
    _bshape = b.shape();
    _oshape = _bshape;
}

void Layer::init_poly(const json& desc)
{
    const auto& coeff = desc["coeff"].get<vector<Scalar>>();
    const auto& ci = _ishape[0];
    const auto& ih = _ishape[1];
    const auto& iw = _ishape[2];
    *_kernel = Tensor<4,Scalar>{3UL, ci, ih, iw};
    auto& w = get<Tensor<4,Scalar>>(*_kernel);
    for (size_t i = 0; i < ci; i++)
        for (size_t ii = 0; ii < ih; ii++)
            for (size_t ij = 0; ij < iw; ij++)
                for (size_t c = 0; c < 3; c++)
                    w[c][i][ii][ij] = coeff[c];
    _kshape = w.shape();
    _stride = {1, 1};
    _padding = {0, 0};
    _oshape = _ishape;
}

// Public functions

const Tensor<3,Scalar>& Layer::bias() const
{
    if (!_bias) throw invalid_argument("ml::Layer::bias: Bias uninitialized");
    return *_bias;
}

const vector<size_t>& Layer::bshape() const
{
    return _bshape;
}

Scalar& Layer::divisor()
{
    return _divisor;
}

const Scalar& Layer::divisor() const
{
    return _divisor;
}

Layer Layer::fuse(const Layer& layer1, const Layer& layer2) // POLY(BATCHNORM2D) -> POLY or BATCHNORM2D(CONV2D) -> CONV2D
{
    if (layer1.type() == LayerType::BATCHNORM2D && layer2.type() == LayerType::POLY) return fuse_batch_poly(layer1, layer2);
    if (layer1.type() == LayerType::CONV2D && layer2.type() == LayerType::BATCHNORM2D) return fuse_conv_batch(layer1, layer2);
    throw invalid_argument("ml::Layer::fuse: Invalid layer types (usage: fuse(batch_layer, poly_layer) or fuse(conv_layer, batch_layer))");
}

Layer Layer::fuse(const Layer& batch1_layer, const Layer& add_layer, const Layer& poly_layer) // POLY(ADD(BATCHNORM2D,BATCHNORM2D)) -> POLYSKIP
{
    if (poly_layer.type() != LayerType::POLY || add_layer.type() != LayerType::ADD || batch1_layer.type() != LayerType::BATCHNORM2D)
        throw invalid_argument("ml::Layer::fuse: Invalid layer types (usage: fuse(poly_layer, add_layer, batch1_layer, batch2_layer))");

    if (poly_layer.oshape() != add_layer.oshape() && poly_layer.oshape() != batch1_layer.oshape())
        throw invalid_argument("ml::Layer::fuse: Layer shapes do not match");

    const auto& a = get<Tensor<4,Scalar>>(*batch1_layer._kernel);
    const auto& c = get<Tensor<4,Scalar>>(*poly_layer._kernel);

    auto shape = c.shape();
    shape[0] = 6;
    Tensor<4,Scalar> d(shape);
    auto c2_a0 = c[2] * a[0];
    auto c2_a1 = c[2] * a[1];
    d[0] = a[0] * (c2_a0 + c[1]) + c[0]; // d0 = c2*a0^2 + c1*a0 + c0 = a0*(c2*a0 + c1) + c0
    d[1] = c2_a0 * 2 + c[1]; // db = 2*c2*a0 + c1
    d[2] = a[1] * d[1]; // da = a1*(2*c2*a0 + c1)
    d[3] = c2_a1 * 2; // dab = 2 * c2 * a1
    d[4] = c[2]; // db2 = c2
    d[5] = c2_a1 * a[1]; // da2 = a1^2 * c2
    return Layer
    (
        LayerType::POLYSKIP, d, poly_layer.bias(), poly_layer.ishape(), shape, poly_layer.bshape(),
        poly_layer.stride(), poly_layer.padding(), batch1_layer.divisor()
    );
}

Layer Layer::fuse(const Layer& batch1_layer, const Layer& batch2_layer, const Layer& add_layer, const Layer& poly_layer) // POLY(ADD(BATCHNORM2D,BATCHNORM2D)) -> POLYSKIP
{
    if (poly_layer.type() != LayerType::POLY || add_layer.type() != LayerType::ADD || batch1_layer.type() != LayerType::BATCHNORM2D || batch2_layer.type() != LayerType::BATCHNORM2D)
        throw invalid_argument("ml::Layer::fuse: Invalid layer types (usage: fuse(poly_layer, add_layer, batch1_layer, batch2_layer))");

    if (poly_layer.oshape() != add_layer.oshape() && poly_layer.oshape() != batch1_layer.oshape() && poly_layer.oshape() != batch2_layer.oshape())
        throw invalid_argument("ml::Layer::fuse: Layer shapes do not match");

    const auto& a = get<Tensor<4,Scalar>>(*batch1_layer._kernel);
    const auto& b = get<Tensor<4,Scalar>>(*batch2_layer._kernel);
    const auto& c = get<Tensor<4,Scalar>>(*poly_layer._kernel);

    auto shape = c.shape();
    shape[0] = 6;
    Tensor<4,Scalar> d(shape);
    auto a0b0 = a[0] + b[0];
    auto c1_2a0b0c2 = a0b0 * c[2] * 2 + c[1];
    d[0] = a0b0 * a0b0 * c[2] + a0b0 * c[1] + c[0]; // d0 = c2*(a0+b0)^2 + c1*(a0+b0) + c0
    d[1] = b[1] * c1_2a0b0c2; // db = b1*(2*c2*(a0+b0) + c1)
    d[2] = a[1] * c1_2a0b0c2; // da = a1*(2*c2*(a0+b0) + c1)
    d[3] = a[1] * b[1] * c[2] * 2; // dab = 2 * c2 * a1 * b1
    d[4] = b[1] * b[1] * c[2]; // db2 = b1^2 * c2
    d[5] = a[1] * a[1] * c[2]; // da2 = a1^2 * c2
    return Layer
    (
        LayerType::POLYSKIP, d, poly_layer.bias(), poly_layer.ishape(), shape, poly_layer.bshape(),
        poly_layer.stride(), poly_layer.padding(), batch1_layer.divisor()
    );
}

Layer Layer::fuse_batch_poly(const Layer& batch_layer, const Layer& poly_layer)
{
    if (batch_layer.oshape() != poly_layer.oshape())
        throw invalid_argument("ml::Layer::fuse: Layer shapes do not match when fusing BATCHNORM2D and POLY layers");

    const auto& b = get<Tensor<4,Scalar>>(*batch_layer._kernel);
    const auto& c = get<Tensor<4,Scalar>>(*poly_layer._kernel);

    Tensor<4,Scalar> d(c.shape());
    auto c2b0 = c[2] * b[0];
    auto c2b0_c1 = c2b0 + c[1];
    d[2] = c[2] * (b[1] * b[1]); // d2 = c2*b1^2
    d[1] = b[1] * (c2b0_c1 + c2b0); // d1 = 2*c2*b0*b1 + c1*b1 = b1*(2*c2*b0 + c1)
    d[0] = b[0] * c2b0_c1 + c[0]; // d0 = c2*b0^2 + c1*b0 + c0 = b0*(c2*b0 + c1) + c0

    return Layer
    (
        LayerType::POLY, d, poly_layer.bias(), poly_layer.ishape(), poly_layer.kshape(), poly_layer.bshape(),
        poly_layer.stride(), poly_layer.padding(), poly_layer.divisor()
    );
}

Layer Layer::fuse_conv_batch(const Layer& conv_layer, const Layer& batch_layer)
{
    if (conv_layer.oshape() != batch_layer.oshape())
        throw invalid_argument("ml::Layer::fuse: Layer shapes do not match when fusing CONV2D and BATCHNORM2D layers");
    
    const auto& b = get<Tensor<4,Scalar>>(*batch_layer._kernel);
    if (!is_coeff_uniform(b, 1)) // check if b[c][i][ii][ij] == [c][i][ji][jj]
        throw invalid_argument("ml::Layer::fuse: BATCHNORM2D kernel is not uniform");

    const auto& c = get<Tensor<6,Scalar>>(*conv_layer._kernel);
    auto co = c.shape()[0];
    Tensor<6,Scalar> d(c.shape()); // {co, kh, kw, ci, ih, iw};
    for (size_t o = 0; o < co; o++) d[o] = c[o] * b[1][o][0][0]; // dk = ck*b1
    const auto& cbias = *conv_layer._bias;
    auto dbias = cbias * b[1] + b[0]; // db = cb*b1 + b0

    return Layer
    (
        LayerType::CONV2D, d, dbias, conv_layer.ishape(), conv_layer.kshape(), conv_layer.bshape(),
        conv_layer.stride(), conv_layer.padding(), batch_layer.divisor()
    );
}

bool Layer::has_updatable_kernel() const
{
    return _type == LayerType::CONV2D || _type == LayerType::LINEAR ||
        (_type == LayerType::BATCHNORM2D && get<Tensor<4,Scalar>>(*_kernel).size() == 2) ||
        (_type == LayerType::POLY        && get<Tensor<4,Scalar>>(*_kernel).size() == 3) ||
        (_type == LayerType::POLYSKIP    && get<Tensor<4,Scalar>>(*_kernel).size() == 6);
}

const vector<size_t>& Layer::ishape() const
{
    return _ishape;
}

variant<Tensor<4,Scalar>,Tensor<6,Scalar>>& Layer::kernel()
{
    if (!_kernel) throw invalid_argument("ml::Layer::kernel: Kernel uninitialized");
    return *_kernel;
}

const variant<Tensor<4,Scalar>,Tensor<6,Scalar>>& Layer::kernel() const
{
    if (!_kernel) throw invalid_argument("ml::Layer::kernel: Kernel uninitialized");
    return *_kernel;
}

const vector<size_t>& Layer::kshape() const
{
    return _kshape;
}


const vector<size_t>& Layer::oshape() const
{
    return _oshape;
}

const vector<size_t>& Layer::padding() const
{
    return _padding;
}

const vector<size_t>& Layer::stride() const
{
    return _stride;
}

const LayerType& Layer::type() const
{
    return _type;
}

void Layer::update_kernel_backward(const Scalar& coeff)
{
    Scalar scale = coeff / _divisor;
    switch (_type)
    {
        case LayerType::BATCHNORM2D:
        case LayerType::POLY:
        case LayerType::POLYSKIP:
        {
            if (!_kernel) throw invalid_argument("ml::Layer::update_kernel_backward: No kernel");
            auto& w = get<Tensor<4,Scalar>>(*_kernel);
            w *= scale;
            break;
        }
        case LayerType::CONV2D:
        {
            if (!_kernel) throw invalid_argument("ml::Layer::update_kernel_backward: No kernel");
            auto& w = get<Tensor<6,Scalar>>(*_kernel);
            w *= scale;
            auto& b = *_bias;
            b *= scale;
            break;
        }
        case LayerType::LINEAR:
        {
            if (!_kernel) throw invalid_argument("ml::Layer::update_kernel_backward: No kernel");
            auto& w = get<Tensor<4,Scalar>>(*_kernel);
            w *= scale;
            auto& b = *_bias;
            b *= scale;
            break;
        }
        default: throw invalid_argument(string("ml::Layer::update_kernel_backward: Cannot update kernel of type LayerType::") + to_string(_type));
    }
    _divisor = 1.0;
}

void Layer::update_kernel_backward(const Tensor<3,Scalar>& coeff)
{
    Tensor<3,Scalar> scale = coeff * (1.0 / _divisor);
    switch (_type)
    {
        case LayerType::BATCHNORM2D:
        case LayerType::POLY:
        case LayerType::POLYSKIP:
        {
            if (!_kernel) throw invalid_argument("ml::Layer::update_kernel_backward: No kernel");
            auto& w = get<Tensor<4,Scalar>>(*_kernel);
            w *= scale;
            break;
        }
        case LayerType::CONV2D:
        {
            if (!_kernel) throw invalid_argument("ml::Layer::update_kernel_backward: No kernel");
            if (!is_coeff_uniform(scale)) throw invalid_argument("ml::Layer::update_kernel_backward: Coefficients are not uniform");
            auto& w = get<Tensor<6,Scalar>>(*_kernel);
            auto co = w.shape()[0];
            for (size_t o = 0; o < co; o++) w[o] *= scale[o][0][0];
            auto& b = *_bias;
            b *= scale;
            break;
        }
        case LayerType::LINEAR:
        {
            if (!_kernel) throw invalid_argument("ml::Layer::update_kernel_backward: No kernel");
            if (!is_coeff_uniform(scale)) throw invalid_argument("ml::Layer::update_kernel_backward: Coefficients are not uniform");
            auto& w = get<Tensor<4,Scalar>>(*_kernel);
            auto co = w.shape()[0];
            for (size_t o = 0; o < co; o++) w[o] *= scale[o][0][0];
            auto& b = *_bias;
            b *= scale;
            break;
        }
        default: throw invalid_argument(string("ml::Layer::update_kernel_backward: Cannot update kernel of type LayerType::") + to_string(_type));
    }
    _divisor = 1.0;
}

void Layer::update_kernel_forward(const Scalar& scale)
{
    vector<Scalar> scales = {scale / _divisor};
    switch (_type)
    {
        case LayerType::BATCHNORM2D:
        case LayerType::POLY:
        {
            if (!_kernel) throw invalid_argument("ml::Layer::update_kernel: No kernel");
            scales.push_back(scales[0] * scales[0]);
            auto& w = get<Tensor<4,Scalar>>(*_kernel);
            for (size_t i = 1; i < w.size(); i++) w[i] *= scales[i - 1];
            break;
        }
        case LayerType::CONV2D:
        {
            if (!_kernel) throw invalid_argument("ml::Layer::update_kernel: No kernel");
            auto& w = get<Tensor<6,Scalar>>(*_kernel);
            w *= scales[0];
            break;
        }
        case LayerType::LINEAR:
        {
            if (!_kernel) throw invalid_argument("ml::Layer::update_kernel: No kernel");
            auto& w = get<Tensor<4,Scalar>>(*_kernel);
            w *= scales[0];
            break;
        }
        case LayerType::POLYSKIP: // assuming the update is in the skip connection
        {
            if (!_kernel) throw invalid_argument("ml::Layer::update_kernel: No kernel");
            scales.push_back(scales[0] * scales[0]);
            auto& w = get<Tensor<4,Scalar>>(*_kernel);
            w[1] *= scales[0]; // db
            w[3] *= scales[0]; // dab
            w[4] *= scales[1]; // db2
            break;
        }
        default: throw invalid_argument(string("ml::Layer::update_kernel: Cannot update kernel of type LayerType::") + to_string(_type));
    }
    _divisor = 1.0;
}

void Layer::update_kernel_forward(const Tensor<3,Scalar>& scale)
{
    vector<Tensor<3,Scalar>> scales = {scale * (1.0 / _divisor)};
    switch (_type)
    {
        case LayerType::BATCHNORM2D:
        case LayerType::POLY:
        {
            if (!_kernel) throw invalid_argument("ml::Layer::update_kernel: No kernel");
            scales.push_back(scales[0] * scales[0]);
            auto& w = get<Tensor<4,Scalar>>(*_kernel);
            for (size_t i = 1; i < w.size(); i++) w[i] *= scales[i - 1];
            break;
        }
        case LayerType::CONV2D:
        {
            if (!_kernel) throw invalid_argument("ml::Layer::update_kernel: No kernel");
            auto& w = get<Tensor<6,Scalar>>(*_kernel);
            w *= scales[0];
            break;
        }
        case LayerType::LINEAR:
        {
            if (!_kernel) throw invalid_argument("ml::Layer::update_kernel: No kernel");
            auto& w = get<Tensor<4,Scalar>>(*_kernel);
            w *= scales[0];
            break;
        }
        case LayerType::POLYSKIP: // assuming the update is in the skip connection
        {
            if (!_kernel) throw invalid_argument("ml::Layer::update_kernel: No kernel");
            scales.push_back(scales[0] * scales[0]);
            auto& w = get<Tensor<4,Scalar>>(*_kernel);
            w[1] *= scales[0]; // db
            w[3] *= scales[0]; // dab
            w[4] *= scales[1]; // db2
            break;
        }
        default: throw invalid_argument(string("ml::Layer::update_kernel: Cannot update kernel of type LayerType::") + to_string(_type));
    }
    _divisor = 1.0;
}

// External functions

LayerType to_layer(const string& layer_name)
{
    if (layer_name == "ADD") return LayerType::ADD;
    if (layer_name == "AVGPOOL2D") return LayerType::AVGPOOL2D;
    if (layer_name == "BATCHNORM2D") return LayerType::BATCHNORM2D;
    if (layer_name == "CONV2D") return LayerType::CONV2D;
    if (layer_name == "LINEAR") return LayerType::LINEAR;
    if (layer_name == "POLY") return LayerType::POLY;
    if (layer_name == "POLYSKIP") return LayerType::POLYSKIP;
    throw invalid_argument("ml::to_layer: Invalid layer name '" + layer_name + "'");
}

} // ml

namespace std
{

const char* to_string(const ml::LayerType& layer, bool pad)
{
    switch (layer)
    {
        case ml::LayerType::ADD:         return pad ? "ADD        " : "ADD";
        case ml::LayerType::AVGPOOL2D:   return pad ? "AVGPOOL2D  " : "AVGPOOL2D";
        case ml::LayerType::BATCHNORM2D: return pad ? "BATCHNORM2D" : "BATCHNORM2D";
        case ml::LayerType::CONV2D:      return pad ? "CONV2D     " : "CONV2D";
        case ml::LayerType::LINEAR:      return pad ? "LINEAR     " : "LINEAR";
        case ml::LayerType::POLY:        return pad ? "POLY       " : "POLY";
        case ml::LayerType::POLYSKIP:    return pad ? "POLYSKIP   " : "POLYSKIP";
        default: throw invalid_argument("ml::to_string");
    }
}

} // std