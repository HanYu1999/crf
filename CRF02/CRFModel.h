#pragma once

#include "Config.h"
#include "DataSet.h"
#include "FactorGraph.h"

// TODO: add an include
#include <cmath>

class EdgeFactorFunction : public FactorFunction {
public:
    int num_label;
    double *lambda;
    map<int, int> *feature_offset;

    // TODO 5
    int num_attr;
    double *combined;

    // TODO 5, change the param
    EdgeFactorFunction(int num_label, int num_attr, double *p_lambda, map<int, int> *edge_feature_offset) {
        this->num_label = num_label;
        this->lambda = p_lambda;
        this->feature_offset = edge_feature_offset;

        // TODO 5
        this->num_attr = num_attr;
        this->combined = new double[num_label * num_label];
    }

    // TODO 5
    void recompute_combined() {
        double res = 0;
        for (int y1 = 0; y1 < num_label; y1++) {
            for (int y2 = y1; y2 < num_label; y2++) {
                res = 0;
                for (int x1 = 0; x1 < num_attr; x1++) {
                    for (int x2 = 0; x2 < num_attr; x2++) {
                        res += lambda[(*feature_offset)[(y1 * num_label + y2) * num_attr * num_attr + x1 * num_attr + x2]];
                    }
                }
                combined[y1 * num_label + y2] = exp(res/num_attr/num_attr);
            }
        }
    }

    virtual double GetValue(int y1, int y2) {
        int a = y1 < y2 ? y1 : y2;
        int b = y1 > y2 ? y1 : y2;
        int i = (*feature_offset)[a * num_label + b];
        //return exp(lambda[i]);
        // TODO 5
        return combined[a * num_label + b];
    }
};

class CRFModel {
public:
    Config *conf;
    DataSet *train_data;
    DataSet *test_data;

    int num_sample;
    int num_label;
    int num_attrib_type;
    int num_edge_type;

    int num_feature;

    // TODO: 1, the length of lambda should be changed
    double *lambda;
    FactorGraph *sample_factor_graph;

    int num_attrib_parameter;
    int num_edge_feature_each_type;
    map<int, int> edge_feature_offset;
    EdgeFactorFunction **func_list;

    // TODO 2, l1
    double alpha_edge, alpha_node;

    CRFModel() {
    }

    void InitTrain(Config *conf, DataSet *train_data);

    void GenFeature();

    void SetupFactorGraphs();

    void Train();

    double CalcGradient(double *gradient);

    double CalcPartialLabeledGradientForSample(DataSample *sample, FactorGraph *factor_graph, double *gradient);

    void SelfEvaluate();

    void PartialSelfEvaluation();

    void InitEvaluate(Config *conf, DataSet *test_data);

    void Evalute();

    int GetAttribParameterId(int y, int x) {
        return y * num_attrib_type + x;
    }

    // TODO: 1, change the GetEdgeParameterId
//    int GetEdgeParameterId(int edge_type, int a, int b) {
//        int offset = edge_feature_offset[(a < b ? a : b) * num_label + (a > b ? a : b)];
//        return num_attrib_parameter + edge_type * num_edge_feature_each_type + offset;
//    }

    int GetEdgeParameterId(int edge_type, int a, int b, int x1, int x2){
        int index1 = (a < b ? a : b) * num_label + (a > b ? a : b);
        int index2 = (a < b ? x1 : x2) * num_attrib_type + (a > b ? x1 : x2);
        int offset = edge_feature_offset[index1 * num_attrib_type * num_attrib_type + index2];
        return num_attrib_parameter + edge_type * num_edge_feature_each_type + offset;
    }

    ~CRFModel() {
        Clean();
    }

    void Clean();

    void SaveModel(const char *file_name);

    void LoadModel(const char *file_name);

    void Estimate(Config *conf);

    void EstimateContinue(Config *conf);

    void Inference(Config *conf);

    int N, M;
    vector<int> state;
    vector<int> best_state;
    vector<int> unlabeled;
    vector<int> test;
    vector<int> valid;

    void LoadStateFile(const char *filename);

    void MHTrain(Config *conf);

    double MHEvaluate(int max_iter = 0, bool isfinal = false);

    void SavePred(const char *filename);

    void MHTrain1(Config *conf);

    bool train1_sample(int type, int center, int ynew, double p, vector<int> &_state, map<int, double> &_gradient);
};
