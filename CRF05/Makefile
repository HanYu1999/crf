maincc: Config.o CRFModel.o DataSet.o FactorGraph.o MHTrain.o MHTrain1.o OpenCRF_Main.o Util.o
	g++ -g -lm -o maincc Config.o CRFModel.o DataSet.o FactorGraph.o MHTrain.o MHTrain1.o OpenCRF_Main.o Util.o

Config.o: Config.cpp Config.h
	g++ -c -g -lm Config.cpp
CRFModel.o: CRFModel.cpp Constant.h CRFModel.h
	g++ -c -g -lm CRFModel.cpp
DataSet.o: DataSet.cpp Constant.h DataSet.h
	g++ -c -g -lm DataSet.cpp
FactorGraph.o: FactorGraph.cpp Constant.h FactorGraph.h
	g++ -c -g -lm FactorGraph.cpp
MHTrain.o: MHTrain.cpp CRFModel.h Constant.h
	g++ -c -g -lm MHTrain.cpp
MHTrain1.o: MHTrain1.cpp CRFModel.h Constant.h
	g++ -c -g -lm MHTrain1.cpp
OpenCRF_Main.o: OpenCRF_Main.cpp Config.h DataSet.h CRFModel.h
	g++ -c -g -lm OpenCRF_Main.cpp
Util.o: Util.cpp Util.h
	g++ -c -g -lm Util.cpp

