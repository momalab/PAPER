# Privacy-Preserving Inference with PAPER

Describe what this directory is about.



# Dependencies

The versions listed below represent the earliest ones on which PAPER inference has been tested.
Testing was performed on Ubuntu 22.04 and Ubuntu 24.04.
It may work with older versions or other environments, but these have not been verified.

* Make $\ge$ 4.3

* CMake $\ge$ 3.22.1

* Git $\ge$ 2.34.1

* GNU C/C++17 compiler (or newer)

* GMP library $\ge$ 6.2.1

* JSON for Modern C++ `nlohmann/json` library $\ge$ 3.10.5

* Python 3 $\ge$ 3.10.12

* NumPY library $\ge$ 1.26.4

```
sudo apt install make cmake git gcc g++ libgmp-dev nlohmann-json3-dev python3 python3-numpy
```

* Microsoft SEAL library 4.1. Check dependencies in https://github.com/microsoft/seal and install with
```
cd third-party
bash install_seal.sh
```

# List of Programs

* `model_plain`: Plaintext single-model runner equivalent to the Python implementation.

* `model_naive`: Privacy-preserving single-model runner using naive encoding.

* `model_adapt`: Privacy-preserving single-model runner using adaptive HW encoding.

* `ensemble_plain`: Plaintext ensemble runner equivalent to the Python implementation.

* `ensemble_adapt`: Privacy-preserving ensemble runner using adaptive HW encoding.

The *ensemble* versions generalize the *model* versions. We keep the model variants since they were already implemented and are easier to follow.
No ensemble version is provided for the *naive* encoding, as it would be inefficient and unnecessary. The naive version exists only to validate the FHE-related classes.

# Compilation

All programs can be compiled using the following command:
```bash
make <program> [PARAMETERS]
```
where `<program>` is the target program name, and `[PARAMETERS]` are optional compilation parameters.

### Table of Parameters

| **Parameter** | **Options** | **Affected Programs** |
|----------------|-------------|------------------------|
| **LIBRARY** | **mockup**, seal | adapt, naive |
| **CONVOLUTION** | classical, map, **mapmem** | adapt |
| **RNS** | **forward**, reverse, seal | adapt, naive |

**Bold** options are default.

### Parameter Descriptions

* **LIBRARY**: Selects the FHE backend.
Use `seal` to enable the Microsoft SEAL library or `mockup` for a lightweight, insecure mock implementation.
The mockup option is significantly faster and intended only for testing or debugging.

* **CONVOLUTION**: Specifies the algorithm used for convolutional layer execution.
    * `classical`: standard convolution algorithm.
    * `map`: optimized for weight clustering.
    * `mapmem`: similar to map, but further optimized for reduced memory usage.

* **RNS**: Determines how the RNS moduli are generated.
    * `seal`: uses the default moduli from the Microsoft SEAL library.
    * `forward` / `reverse`: use custom algorithms to define the moduli in forward or reverse order, respectively.

### Examples

* `make ensemble_plain` compiles the `ensemble_plain` program.
* `make ensemble_adapt LIBRARY=seal` compiles `ensemble_adapt` using the Microsoft SEAL library for FHE operations, the default convolution algorithm (`mapmem`), and custom forward-mode RNS moduli generation.

# Run

...

