#pragma once

#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "encodings.h"
#include "layer.h"
#include "model_base.h"
#include "scalar.h"
#include "tensor.h"

namespace fhe
{

namespace plain
{

class Model : ml::BaseModel
{
    private:
        void compute(std::size_t, std::unordered_set<std::size_t>&, std::unordered_map<std::size_t,type::Tensor<3,Scalar>>&) const;

    public:
        Model() = default;
        Model(const Model&) = default;
        Model(const nlohmann::json&, const std::vector<std::size_t>&);
        Model(std::ifstream, const std::vector<std::size_t>&);
        Model(const std::string&, const std::vector<std::size_t>&);
        Model(Model&&) = default;
        ~Model() = default;

        Model& operator=(const Model&) = default;
        Model& operator=(Model&&) = default;

        type::Tensor<1,double> classify(const type::Tensor<3,Scalar>& decoded_output) const;
        type::Tensor<3,Scalar> decode_output(const type::Tensor<3,Scalar>& output) const;
        type::Tensor<3,Scalar> encode_input(const type::Tensor<3,Scalar>& input) const;
        virtual Encoding encoding() const override;
        type::Tensor<3,Scalar> forward(type::Tensor<3,Scalar>& input) const;
        type::Tensor<3,Scalar> forward(const type::Tensor<3,Scalar>& input) const;
        void print() const;
};

} // plain

} // fhe