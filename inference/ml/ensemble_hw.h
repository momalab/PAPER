#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "ciphertext.h"
#include "encode.h"
#include "layer.h"
#include "model_base.h"
#include "tensor.h"

namespace fhe
{

namespace hw
{

class Ensemble
{
    private:
        std::vector<ml::BaseModel> models; // models in the ensemble
        std::vector<int> offsets; // offsets for ensemble
        std::vector<std::variant<type::Tensor<5,Scalar>, type::Tensor<7,Scalar>>> kernels; // combined kernels for all models
        std::vector<type::Tensor<4,Scalar>> biases; // combined biases for all models
        std::vector<Configuration> configs;
        std::vector<int> in_depths; // multiplicative depth of each node's input
        std::vector<int> out_depths; // multiplicative depth of each node's output
        std::size_t first_id, last_id;

        void compute(std::size_t, std::unordered_set<std::size_t>&, std::unordered_map<std::size_t,std::shared_ptr<type::Tensor<2,Ciphertext>>>&) const;
        void config(std::size_t, std::unordered_set<std::size_t>&);
        void init();

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

        type::Tensor<1,double> classify(const type::Tensor<4,Scalar>& decoded_output) const;
        type::Tensor<4,Scalar> decode_output(const type::Tensor<2,Ciphertext>& output) const;
        type::Tensor<2,Ciphertext> encode_input(const type::Tensor<3,Scalar>& input) const;
        Encoding encoding() const;
        type::Tensor<2,Ciphertext> forward(const type::Tensor<2,Ciphertext>& input) const;
        void print() const;
};

} // hw

} // fhe