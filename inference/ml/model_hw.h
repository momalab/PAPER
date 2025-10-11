#pragma once

#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
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

class Model : ml::BaseModel
{
    private:
        std::vector<Configuration> configs;
        std::vector<int> in_depths; // multiplicative depth of each node's input
        std::vector<int> out_depths; // multiplicative depth of each node's output
        std::vector<std::variant<type::Tensor<5,Plaintext>,type::Tensor<3,Plaintext>>> kernels;
        std::vector<type::Tensor<2,Plaintext>> biases;
        std::vector<std::vector<std::unordered_map<Scalar,type::Tensor<1,Plaintext>>>> kernel_maps; // FIXME test remove
        std::size_t first_id;

        void compute(std::size_t, std::unordered_set<std::size_t>&, std::unordered_map<std::size_t,std::shared_ptr<type::Tensor<2,Ciphertext>>>&) const;
        void config(std::size_t, std::unordered_set<std::size_t>&);
        void encode_biases();
        void encode_kernels();
        void init();
        void preencode_kernels();

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
        type::Tensor<3,Scalar> decode_output(const type::Tensor<2,Ciphertext>& output) const;
        type::Tensor<2,Ciphertext> encode_input(const type::Tensor<3,Scalar>& input) const;
        virtual Encoding encoding() const override;
        type::Tensor<2,Ciphertext> forward(type::Tensor<2,Ciphertext>& input) const;
        type::Tensor<2,Ciphertext> forward(const type::Tensor<2,Ciphertext>& input) const;
        void print() const;
};

} // hw

} // fhe