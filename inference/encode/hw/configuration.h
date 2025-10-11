#pragma once

#include <vector>
#include "layer.h"
#include "tensor.h"

namespace fhe
{

namespace hw
{

class Configuration // Mapping
{
    private:
        std::vector<type::Tensor<2,std::size_t>> mappings; // Ih x Iw
        std::vector<ml::Layer> layers;
        std::vector<std::size_t> nCTs;
        std::size_t index;

        void init_mapping(const type::Tensor<2,std::size_t>&);
        void init_nCT(std::size_t = 0);
        void remap(std::size_t index);
        void set_input_layer(const std::vector<std::size_t>& ishape);

    public:
        Configuration() = default;
        Configuration(const Configuration&) = default;
        Configuration(Configuration&&) = default;
        ~Configuration() = default;
        Configuration(const std::vector<ml::Layer>& layers);
        Configuration(const std::vector<ml::Layer>& layers, type::Tensor<2,std::size_t> initial_mapping);
        Configuration(const std::vector<ml::Layer>& layers, const std::vector<std::size_t>& padding);
        Configuration(const std::vector<ml::Layer>& layers, type::Tensor<2,std::size_t> initial_mapping, std::size_t initial_nCTs);
        Configuration(const std::vector<ml::Layer>& layers, type::Tensor<2,std::size_t> initial_mapping, std::size_t initial_nCTs, const std::vector<std::size_t>& padding);

        Configuration& operator=(const Configuration&) = default;
        Configuration& operator=(Configuration&&) = default;

        const ml::LayerType& type() const;
        const std::vector<std::size_t>& ishape() const;
        const std::vector<std::size_t>& kshape() const;
        const std::vector<std::size_t>& oshape() const;
        const std::vector<std::size_t>& stride() const;
        const std::vector<std::size_t>& padding() const;
        fhe::Scalar divisor() const;
        std::vector<std::size_t> max_padding() const;
        const type::Tensor<2,std::size_t>& mapping() const;
        const type::Tensor<2,std::size_t>& imap() const;
        const type::Tensor<2,std::size_t>& omap() const;
        const std::size_t& nct() const;
        const std::size_t& ncti() const;
        const std::size_t& ncto() const;

        // functions to move the head
        std::size_t get() const;
        void next();
        void previous();
        void reset();
        void set(std::size_t);

        // change the configuration
        void push_back(const ml::Layer&);
        void pop_back();

        // auxiliary functions
        int shift(std::size_t krow, std::size_t kcol) const;
};

} // hw

} // fhe