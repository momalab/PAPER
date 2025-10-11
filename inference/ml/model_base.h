#pragma once

#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_set>
#include <vector>
#include "encodings.h"
#include "layer.h"
#include "tensor.h"

namespace ml
{

class BaseModel
{
    private:
        void fuse_nodes();
        void fuse_conv_batch();
        void fuse_poly_add_batch();
        void fuse_poly_batch();
        void handle_avgpool2d(std::size_t);
        void init(const nlohmann::json&, const std::vector<std::size_t>&);
        bool is_solvable_backward(std::size_t, std::size_t) const;
        bool is_solvable_bypass(std::size_t, std::size_t) const;
        bool is_solvable_forward(std::size_t) const;
        void redistribute_weights(std::size_t, std::unordered_set<std::size_t>&);
        void remove_isolated_nodes();
        void solve(std::size_t);
        void update_kernel_backward(std::size_t, std::size_t, const fhe::Scalar&);
        void update_kernel_backward(std::size_t, std::size_t, const type::Tensor<3,fhe::Scalar>&);
        void update_kernel_bypass(std::size_t, std::size_t, const fhe::Scalar&);
        void update_kernel_bypass(std::size_t, std::size_t, const type::Tensor<3,fhe::Scalar>&);
        void update_kernel_forward(std::size_t, const fhe::Scalar&);
        void update_kernel_forward(std::size_t, const type::Tensor<3,fhe::Scalar>&);

    protected:
        struct Node
        {
            Layer layer;
            std::vector<std::size_t> previous;
            std::vector<std::size_t> next;
        };
        std::vector<Node> nodes;
        std::size_t last_id;

    public:
        BaseModel() = default;
        BaseModel(const BaseModel&) = default;
        BaseModel(const nlohmann::json&, const std::vector<std::size_t>&);
        BaseModel(std::ifstream, const std::vector<std::size_t>&);
        BaseModel(const std::string&, const std::vector<std::size_t>&);
        BaseModel(BaseModel&&) = default;
        ~BaseModel() = default;

        BaseModel& operator=(const BaseModel&) = default;
        BaseModel& operator=(BaseModel&&) = default;

        const std::vector<Node>& get_nodes() const;
        std::size_t get_last_id() const;

        virtual fhe::Encoding encoding() const;
        void print() const;
};

} // ml