ROOT=$(abspath ../..)
3P=$(ROOT)/3p
SRC=$(ROOT)/inference
CFG=$(SRC)/config
ENC=$(SRC)/encode
FHE=$(SRC)/fhe
FHE_MOCKUP=$(FHE)/mockup
FHE_SEAL=$(FHE)/seal
ML=$(SRC)/ml
TYPE=$(SRC)/type
UTIL=$(SRC)/util
SEAL_INC=$(3P)/seal/include/SEAL-4.1
SEAL_LIB=$(3P)/seal/lib

CC=g++
INCS=-I$(CFG) -I$(ENC) -I$(FHE) -I$(ML) -I$(SEAL_INC) -I$(TYPE) -I$(UTIL)
FLAGS=-O3 -std=c++20 -Wall -Wextra #-Werror
CPPS=\
 $(ENC)/naive.cpp $(ENC)/hw/common.cpp $(ENC)/hw/configuration.cpp $(ENC)/hw/encode.cpp\
 $(FHE)/ciphertext.cpp $(FHE)/plaintext.cpp\
 $(ML)/layer.cpp $(ML)/model_base.cpp $(ML)/model_plain.cpp $(ML)/model_naive.cpp $(ML)/model_hw.cpp\
 $(ML)/ml_plain.cpp $(ML)/ml_naive.cpp $(ML)/hw/adapt.cpp\
 $(ML)/ensemble_plain.cpp $(ML)/ensemble_hw.cpp\
 $(TYPE)/tensor1d.cpp $(TYPE)/tensor1f.cpp\
 $(UTIL)/hash.cpp $(UTIL)/math.cpp $(UTIL)/random.cpp $(UTIL)/text.cpp
LIBS=-lgmp -lgmpxx -pthread -L$(SEAL_LIB) -lseal-4.1
DEFS=

ifeq ($(LIBRARY),seal)
	DEFS+=-DSEAL
	CPPS+=$(FHE_SEAL)/keys.cpp
else
	CPPS+=$(FHE_MOCKUP)/keys.cpp $(FHE_MOCKUP)/native.cpp
endif

ifeq ($(CONVOLUTION),classical)
	DEFS+=-DCONVOLUTION_CLASSICAL
else ifeq ($(CONVOLUTION),map)
	DEFS+=-DCONVOLUTION_MAP
endif

ifeq ($(RNS),seal)
	DEFS+=-DRNS_SEAL
else ifeq ($(RNS),reverse)
	DEFS+=-DRNS_REVERSE
else
	DEFS+=-DRNS_FORWARD
endif

info:
	@echo "make filename"

%: %.cpp
	$(CC) $(FLAGS) $(INCS) -o $@.exe $< $(CPPS) $(LIBS) $(DEFS)

clean:
	rm -f *.exe

resnet18:
	./ensemble_adapt.exe $(MODEL1) $(MODEL2) $(MODEL3) $(MODEL4) $(MODEL5) $(MODEL6) $(MODEL7) $(MODEL8) $(DATASET) 15 22 18x44,1x23,1x54 1,2,4,8,16,24,31,32,33,62,64,66,124,128,132,248,256,264,272,280,512,520,528,536,768,776,784,792,16120,16128,16136,16252,16256,16260,16318,16320,16322,16351,16352,16353,16376,16380,16382,16383 $(NITERS)

resnet20:
	./ensemble_adapt.exe $(MODEL1) $(MODEL2) $(MODEL3) $(MODEL4) $(MODEL5) $(MODEL6) $(MODEL7) $(MODEL8) $(DATASET) 15 21 20x42,1x22,1x44 1,2,4,8,12,16,31,32,33,62,64,66,124,128,132,136,140,256,260,264,268,384,388,392,396,512,528,16252,16256,16260,16318,16320,16322,16351,16352,16353,16380,16382,16383 $(NITERS)

resnet32:
	./ensemble_adapt.exe $(MODEL1) $(MODEL2) $(MODEL3) $(MODEL4) $(MODEL5) $(MODEL6) $(MODEL7) $(MODEL8) $(DATASET) 16 26 32x52,1x27,1x54 1,2,4,8,12,16,31,32,33,62,64,66,124,128,132,136,140,256,260,264,268,384,388,392,396,512,528,32636,32640,32644,32702,32704,32706,32735,32736,32737,32764,32766,32767 $(NITERS)
