maincc: Config.o CRFModel.o DataSet.o FactorGraph.o MHTrain.o MHTrain1.o OpenCRF_Main.o Util.o
	g++ -o maincc Config.o CRFModel.o DataSet.o FactorGraph.o MHTrain.o MHTrain1.o OpenCRF_Main.o Util.o

Config.o: Config.cpp Config.h
	g++ -c Config.cpp
CRFModel.o: CRFModel.cpp Constant.h CRFModel.h
	g++ -c CRFModel.cpp
DataSet.o: DataSet.cpp Constant.h DataSet.h
	g++ -c DataSet.cpp
FactorGraph.o: FactorGraph.cpp Constant.h FactorGraph.h
	g++ -c FactorGraph.cpp
MHTrain.o: MHTrain.cpp CRFModel.h Constant.h
	g++ -c MHTrain.cpp
MHTrain1.o: MHTrain1.cpp CRFModel.h Constant.h
	g++ -c MHTrain1.cpp
OpenCRF_Main.o: OpenCRF_Main.cpp Config.h DataSet.h CRFModel.h
	g++ -c OpenCRF_Main.cpp
Util.o: Util.cpp Util.h
	g++ -c Util.cpp

