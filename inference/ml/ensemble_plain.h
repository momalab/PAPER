#pragma once

#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include "encodings.h"
#include "layer.h"
#include "model_plain.h"
#include "scalar.h"
#include "tensor.h"

namespace fhe
{

namespace plain
{

class Ensemble
{
    private:
        std::vector<Model> models;

    public:
        Ensemble() = default;
        Ensemble(const Ensemble&) = default;
        Ensemble(Ensemble&&) = default;
        Ensemble(const std::vector<nlohmann::json>&, const std::vector<std::size_t>&);
        Ensemble(std::vector<std::ifstream>, const std::vector<std::size_t>&);
        Ensemble(const std::vector<std::string>&, const std::vector<std::size_t>&);
        ~Ensemble() = default;

        Ensemble& operator=(const Ensemble&) = default;
        Ensemble& operator=(Ensemble&&) = default;

        type::Tensor<1,double> classify(const type::Tensor<3,Scalar>& decoded_output) const;
        type::Tensor<3,Scalar> decode_output(const type::Tensor<3,Scalar>& output) const;
        type::Tensor<3,Scalar> encode_input(const type::Tensor<3,Scalar>& input) const;
        Encoding encoding() const;
        type::Tensor<3,Scalar> forward(const type::Tensor<3,Scalar>& input) const;
        void print() const;
};

} // plain

} // fhe