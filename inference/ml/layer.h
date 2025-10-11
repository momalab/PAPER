#pragma once

#include <cstddef>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <variant>
#include <vector>
#include "tensor.h"
#include "scalar.h"

namespace ml
{

enum class LayerType {ADD, AVGPOOL2D, BATCHNORM2D, CONV2D, LINEAR, POLY, POLYSKIP};

class Layer
{
    private:
        LayerType _type;
        std::shared_ptr<std::variant<type::Tensor<4,fhe::Scalar>,type::Tensor<6,fhe::Scalar>>> _kernel;
        std::shared_ptr<type::Tensor<3,fhe::Scalar>> _bias;
        std::vector<std::size_t> _ishape;
        std::vector<std::size_t> _kshape;
        std::vector<std::size_t> _bshape;
        std::vector<std::size_t> _oshape;
        std::vector<std::size_t> _stride;
        std::vector<std::size_t> _padding;
        fhe::Scalar _divisor;

        void init_add();
        void init_avgpool2d(const nlohmann::json&);
        void init_batchnorm(const nlohmann::json&);
        void init_conv2d(const nlohmann::json&);
        void init_linear(const nlohmann::json&);
        void init_poly(const nlohmann::json&);

        static Layer fuse_batch_poly(const Layer&, const Layer&); // POLY(BATCHNORM2D) -> POLY
        static Layer fuse_conv_batch(const Layer&, const Layer&); // BATCHNORM2D(CONV2D) -> CONV2D

    public:
        Layer() = default;
        Layer(const Layer&) = default;
        Layer(Layer&&) = default;
        Layer(const nlohmann::json&, const std::vector<std::size_t>&);
        Layer
        (
            const LayerType&, const std::variant<type::Tensor<4,fhe::Scalar>,type::Tensor<6,fhe::Scalar>>&, const type::Tensor<3,fhe::Scalar>&,
            const std::vector<std::size_t>&, const std::vector<std::size_t>&, const std::vector<std::size_t>&,
            const std::vector<std::size_t>&, const std::vector<std::size_t>&, const fhe::Scalar&
        );
        ~Layer() = default;

        Layer& operator=(const Layer&) = default;
        Layer& operator=(Layer&&) = default;

        const std::vector<std::size_t>& bshape() const;
        const std::vector<std::size_t>& ishape() const;
        const std::vector<std::size_t>& kshape() const;
        const std::vector<std::size_t>& oshape() const;

        const type::Tensor<3,fhe::Scalar>& bias() const;
        fhe::Scalar& divisor();
        const fhe::Scalar& divisor() const;
        std::variant<type::Tensor<4,fhe::Scalar>,type::Tensor<6,fhe::Scalar>>& kernel();
        const std::variant<type::Tensor<4,fhe::Scalar>,type::Tensor<6,fhe::Scalar>>& kernel() const;
        const std::vector<std::size_t>& padding() const;
        const std::vector<std::size_t>& stride() const;
        const LayerType& type() const;
        void update_kernel_backward(const fhe::Scalar&);
        void update_kernel_backward(const type::Tensor<3,fhe::Scalar>&);
        void update_kernel_forward(const fhe::Scalar&);
        void update_kernel_forward(const type::Tensor<3,fhe::Scalar>&);

        bool has_updatable_kernel() const;
        
        static Layer fuse(const Layer&, const Layer&); // POLY(BATCHNORM2D) -> POLY or BATCHNORM2D(CONV2D) -> CONV2D
        static Layer fuse(const Layer&, const Layer&, const Layer&); // POLY(ADD(BATCHNORM2D,I)) -> POLYSKIP
        static Layer fuse(const Layer&, const Layer&, const Layer&, const Layer&); // POLY(ADD(BATCHNORM2D,BATCHNORM2D)) -> POLYSKIP
};

LayerType to_layer(const std::string&);

} // ml

namespace std
{

const char* to_string(const ml::LayerType& layer, bool = false);

} // std