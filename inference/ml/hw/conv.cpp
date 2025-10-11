#ifdef CONVOLUTION_CLASSICAL
#include "conv_classical.cpp"
#elif defined(CONVOLUTIONAL_MAP)
#include "conv_map.cpp"
#else
#include "conv_mapmem.cpp"
#endif