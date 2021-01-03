#pragma once

#include <math.h>
#include "Config.h"
#include "DataSet.h"
#include "FactorGraph.h"

#include <set>

using std::set;

class EdgeFactorFunction: public FactorFunction
{
public:
	int     num_label;
	double* lambda;
	map<int, int>* feature_offset;

	EdgeFactorFunction(int num_label, double* p_lambda, map<int,int>* edge_feature_offset)
	{
		this->num_label = num_label;
		this->lambda = p_lambda;
		this->feature_offset = edge_feature_offset;
	}
	
	virtual double GetValue(int y1, int y2)
	{
		int a = y1 < y2 ? y1 : y2;
		int b = y1 > y2 ? y1 : y2;
		int i = (*feature_offset)[a * num_label + b];
		return exp (lambda[i]);
	}
};

class CRFModel
{
public:
	Config*     conf;
	DataSet*    train_data;
	DataSet*    test_data;

	// TODO anchor node
	set<int> anchor_nodes;

	int         num_sample;
	int         num_label;
	int         num_attrib_type;
	int         num_edge_type;
	
	int         num_feature;

	double      *lambda;
	FactorGraph *sample_factor_graph;

	int             num_attrib_parameter;
	int             num_edge_feature_each_type;
	map<int, int>   edge_feature_offset;
	EdgeFactorFunction**   func_list;

	CRFModel()
	{
	}

	void InitTrain(Config* conf, DataSet* train_data);
	void GenFeature();
	void SetupFactorGraphs();
	// TODO analyse
	void Train(bool is_write_analyse);
	// TODO analyse
	double CalcGradient(double* gradient, bool is_write_analyse);
	// TODO analyse
	double CalcPartialLabeledGradientForSample(DataSample* sample, FactorGraph* factor_graph, double* gradient, bool is_write_analyse);
	// TODO server
	void AnalyseLogic(DataSample* sample, FactorGraph* factor_graph, int* inf_label, double** label_prob, FILE* ana_out, FILE* compare_out, FILE * graph_out);
	// TODO compare
	void CalcLinearBPEquation(DataSample* sample, FactorGraph* factor_graph, int node_id, double** label_prob, double* global_edge_factor, int* global_edge_num, FILE * compare_out, FILE * graph_out);

	int FindAnchorNode(DataSample *sample, FactorGraph *factor_graph, double ** label_prob);
	int FindAnchorNodeV1(DataSample *sample, FactorGraph *factor_graph, double **label_prob);
	int FindAnchorNodeV2(DataSample *sample, FactorGraph *factor_graph, double **label_prob);
	int FindAnchorNodeBound(DataSample *sample, FactorGraph * factor_graph);

	void SelfEvaluate();
	void PartialSelfEvaluation();
	
	void InitEvaluate(Config* conf, DataSet* test_data);
	void Evalute();

	int GetAttribParameterId(int y, int x)
	{
		return y * num_attrib_type + x;
	}

	int GetEdgeParameterId(int edge_type, int a, int b)
	{ 
		int offset = edge_feature_offset[(a<b?a:b) * num_label + (a>b?a:b)];
		return num_attrib_parameter + edge_type * num_edge_feature_each_type + offset;
	}

    ~CRFModel()
	{
		Clean();
	}

	void Clean();

	void SaveModel(const char* file_name);
	void LoadModel(const char* file_name);

	void Estimate(Config* conf);
	void EstimateContinue(Config* conf);
	void Inference(Config* conf);

	int         N, M;
	vector<int> state;
	vector<int> best_state;
	vector<int> unlabeled;
	vector<int> test;
	vector<int> valid;

	void LoadStateFile(const char* filename);
	void MHTrain(Config *conf);
	double MHEvaluate(int max_iter = 0, bool isfinal = false);
	void SavePred(const char* filename);

	void MHTrain1(Config *conf);
	bool train1_sample(int type, int center, int ynew, double p, vector<int>& _state, map<int,double>& _gradient);
};
