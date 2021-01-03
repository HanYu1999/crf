#include "CRFModel.h"
#include "Constant.h"
#include "Util.h"
#include <ctime>
#include <math.h>
#include <string.h>

#define MAX_BUF_SIZE 65536
#define MINI_CHANGE 10e-6

// TODO server
#define KL_DIV(a, b, N, res) \
        res = 0.0; \
        for(int i = 0; i < N; i++) \
            res += a[i] * (log(a[i]) - log(b[i]));

#define BI_KL_DIV(a, b, N, res) \
        double temp1 = 0.0, temp2 = 0.0; \
        KL_DIV(a, b, N, temp1); \
        KL_DIV(b, a, N, temp2); \
        res = temp1 + temp2;

// TODO improve
// TODO make sure the a is distribution
#define ENTROPY(a, N, res) \
		res = 0.0; \
		for(int i = 0; i < N; i++) {\
            if(a[i] == 0) \
				continue; \
            res += - a[i] * log2(a[i]);\
        }

void CRFModel::InitTrain(Config* conf, DataSet* train_data)
{
	this->conf = conf;
	this->train_data = train_data;

	num_sample = train_data->num_sample;
	num_label = train_data->num_label;
	num_attrib_type = train_data->num_attrib_type;
	num_edge_type = train_data->num_edge_type;

	GenFeature();
	lambda = new double[num_feature];
	// Initialize parameters
	for (int i = 0; i < num_feature; i++)
		lambda[i] = 0.0;
	SetupFactorGraphs();

	N = train_data->sample[0]->num_node;
	M = train_data->sample[0]->num_edge;
}

void CRFModel::GenFeature()
{
	num_feature = 0;

	// state feature: f(y, x)
	num_attrib_parameter = num_label * num_attrib_type;
	num_feature += num_attrib_parameter;

	// edge feature: f(edge_type, y1, y2)
	edge_feature_offset.clear();
	int offset = 0;
	for (int y1 = 0; y1 < num_label; y1++)
		for (int y2 = y1; y2 < num_label; y2++)
		{
			edge_feature_offset.insert( make_pair(y1 * num_label + y2, offset) );
			offset ++;
		}
	num_edge_feature_each_type = offset;
	num_feature += num_edge_type * num_edge_feature_each_type;
}

void CRFModel::SetupFactorGraphs()
{
	double* p_lambda = lambda + num_attrib_parameter;
	func_list = new EdgeFactorFunction*[num_edge_type];
	for (int i = 0; i < num_edge_type; i++)
	{
		func_list[i] = new EdgeFactorFunction(num_label, p_lambda, &edge_feature_offset);
		p_lambda += num_edge_feature_each_type;
	}

	sample_factor_graph = new FactorGraph[num_sample];
	for (int s = 0; s < num_sample; s++)
	{
		DataSample* sample = train_data->sample[s];
		
		int n = sample->num_node;
		int m = sample->num_edge;

		sample_factor_graph[s].InitGraph(n, m, num_label);

		// Add node info
		for (int i = 0; i < n; i++)
		{
			sample_factor_graph[s].SetVariableLabel(i, sample->node[i]->label);
			sample_factor_graph[s].var_node[i].label_type = sample->node[i]->label_type;
		}

		// Add edge info
		for (int i = 0; i < m; i++)
		{
            // TODO server
			sample_factor_graph[s].AddEdge(sample->edge[i]->a, sample->edge[i]->b, func_list[sample->edge[i]->edge_type], sample->edge[i]->edge_type);
		}

		sample_factor_graph[s].GenPropagateOrder();
	}
}

// TODO analyse
void CRFModel::Train(bool is_write_analyse)
{    
	double* gradient;
	double  f;          // log-likelihood

	gradient = new double[num_feature + 1];

	///// Initilize all info

	// Data Varible         
	double  old_f = 0.0;

	// Variable for optimization
	int     m_correlation = 3;
	double* work_space = new double[num_feature * (2 * m_correlation + 1) + 2 * m_correlation];
	int     diagco = 0;
	double* diag = new double[num_feature];
	int     iprint[2] = {-1, 0}; // do not print anything
	double  eps = conf->eps;
	double  xtol = 1.0e-16;
	int     iflag = 0;

	// Other Variables
	int     num_iter;
	double  *tmp_store = new double[num_feature + 1];

	// Main-loop of CRF
	// Paramater estimation via Gradient Descend
	num_iter = 0;

	double start_time, end_time;

	do {
		num_iter++;
		start_time = clock();
		

		// TODO normalize the compatibility matrix
                // TODO only deals with the 2 x 2
                for (int t = 0; t < num_edge_type; t ++){
                        double unnorm_v1 = (exp(lambda[GetEdgeParameterId(t, 0, 0)]) + exp(lambda[GetEdgeParameterId(t, 1, 1)]))/2.0;
                        double unnorm_v2 = exp(lambda[GetEdgeParameterId(t, 0, 1)]);
                        double norm_v1 = unnorm_v1 / (unnorm_v1 + unnorm_v2);
                        double norm_v2 = unnorm_v2 / (unnorm_v1 + unnorm_v2);
                        lambda[GetEdgeParameterId(t, 0, 0)] = log(norm_v1);
                        lambda[GetEdgeParameterId(t, 1, 1)] = log(norm_v1);
                        lambda[GetEdgeParameterId(t, 0, 1)] = log(norm_v2);
                }

		// Step A. Calc gradient and log-likehood of the local datas
		// TODO analyse
		f = CalcGradient(gradient, is_write_analyse);

		// Step B. Opitmization by Gradient Descend
		printf("[Iter %3d] log-likelihood : %.8lf\n", num_iter, f);
		fflush(stdout);

		// If diff of log-likelihood is small enough, break.
		if (fabs(old_f - f) < eps) break;
		old_f = f;

		// Normalize Graident
		double g_norm = 0.0;
		for (int i = 0; i < num_feature; i++)
			g_norm += gradient[i] * gradient[i];
		g_norm = sqrt(g_norm);
			
		if (g_norm > 1e-8)
		{
			for (int i = 0; i < num_feature; i++)
				gradient[i] /= g_norm;
		}

		for (int i = 0; i < num_feature; i++)
			lambda[i] += gradient[i] * conf->gradient_step;
		iflag = 1;
		

		if (num_iter % conf->eval_interval == 0)
		{
			SelfEvaluate();
		}

		end_time = clock();

		FILE* ftime = fopen((conf->out_dir + "/time.out").c_str(), "a");
		fprintf(ftime, "start_time = %.6lf\n", start_time);
		fprintf(ftime, "end_time = %.6lf\n", end_time);
		fprintf(ftime, "cost = %.6lf\n", end_time - start_time);
		
		fclose(ftime);

		printf("!!! Time cost = %.6lf\n", end_time - start_time);
		fflush(stdout);

	} while (iflag != 0 && num_iter < conf->max_iter);

	delete[] tmp_store;

	delete[] work_space;
	delete[] diag;

	delete[] gradient;
}

// TODO analyse
double CRFModel::CalcGradient(double* gradient, bool is_write_analyse)
{
	double  f;
	
	// Initialize

	f = 0.0;
	for (int i = 0; i < num_feature; i++)
	{
		gradient[i] = 0;
	}

	// Calculation
	for (int i = 0; i < num_sample; i++)
	{
		// TODO analyse
		double t = CalcPartialLabeledGradientForSample(train_data->sample[i], &sample_factor_graph[i], gradient, is_write_analyse);
		f += t;		 
	}
	
	return f;
}

//// TODO analyse
//void CRFModel::AnalyseLogic(DataSample* sample, FactorGraph* factor_graph, int* inf_label_state, int* inf_label, int index, FILE* ana_out){
//    int neighbor_id = 0;
//    string judgement;
//
//    if (sample->node[index]->label_type == Enum::KNOWN_LABEL) return;
//    if (inf_label[index] == sample->node[index]->label && inf_label_state[index] == sample->node[index]->label) return;
//    if (inf_label[index] != sample->node[index]->label && inf_label_state[index] != sample->node[index]->label) return;
//    if (inf_label[index] == sample->node[index]->label && inf_label_state[index] != sample->node[index]->label)
//        // fprintf(ana_out, "The edge corrects this node: %d\n", (factor_graph->var_node[index]).id);
//	return;
//    else
//        fprintf(ana_out, "The edge mistakes this node: %d\n", (factor_graph->var_node[index]).id);
//
//    fprintf(ana_out, "       The state factor: %f -- %f    The label: %d\n",
//            factor_graph->var_node[index].state_factor[0], factor_graph->var_node[index].state_factor[1],
//            sample->node[index]->label);
//
//    int node_label = sample->node[index]->label;
//    for (int k = 0; k < factor_graph->var_node[index].neighbor.size(); k++) {
//        neighbor_id = (factor_graph->var_node[index].neighbor[k]->neighbor[0])->id;
//        neighbor_id = (neighbor_id == factor_graph->var_node[index].id)
//                      ? (factor_graph->var_node[index].neighbor[k]->neighbor[1])->id : neighbor_id;
//        if (factor_graph->var_node[index].belief[k][node_label] > factor_graph->var_node[index].belief[k][1- node_label]) // Only for the binary case
//            judgement = "correct";
//        else if (inf_label_state[neighbor_id] != sample->node[neighbor_id]->label)
//            judgement = "the prior cause the error";
//        else
//            judgement = "the compatibility matrix cause the error";
//        fprintf(ana_out, "       Neighbor: %d's belief: %f -- %f   label: %d   state factor: %f -- %f   type: %s   %s\n", neighbor_id,
//                factor_graph->var_node[index].belief[k][0], factor_graph->var_node[index].belief[k][1],
//                sample->node[neighbor_id]->label, factor_graph->var_node[neighbor_id].state_factor[0], factor_graph->var_node[neighbor_id].state_factor[1],
//                factor_graph->var_node[neighbor_id].label_type == Enum::KNOWN_LABEL ? "KNOWN": "UNKNOWN", judgement.c_str());
//    }
//}

// TODO server
void CRFModel::AnalyseLogic(DataSample* sample, FactorGraph* factor_graph, int* inf_label, double** label_prob, FILE* ana_out, FILE* compare_out, FILE * graph_out) {
    int n = sample->num_node;
    IdKlDict kl_dict;
    double prior[num_label], post[num_label], msg[num_label], sub_msg[num_label];
    double edge_factor[num_edge_type] = {0.0};
    int edge_num[num_edge_type] = {0};
    // calculate the KL divergence
    for (int i = 0; i < n; i++){
        double cur_kl = 0.0;
        double sum = 0.0;
        if (inf_label[i] == sample->node[i]->label) continue;
        if (sample->node[i]->type1 != Enum1::VALID) continue;
        for (int k = 0; k < num_label; k++)
            sum += factor_graph->var_node[i].state_factor[k];
        for (int k = 0; k < num_label; k++){
            prior[k] = factor_graph->var_node[i].state_factor[k]/sum;
            post[k] = label_prob[k][i];
        }
        BI_KL_DIV(prior, post, num_label, cur_kl);
        kl_dict.nodes.push_back(new NodeKL(i, cur_kl));
    }
    // sort
    kl_dict.sort_id_by_value(true);

    printf("%d\n", conf->is_priorkl_bigger_than_msgkl);
    fflush(stdout);

    int num_debug_node = (int)kl_dict.nodes.size();
    // TODO improve
    double total_entropy[num_edge_type + 1];
    int total_entropy_num[num_edge_type + 1];

    for(int k = 0; k < num_edge_type + 1; k ++){
	    total_entropy[k] = 0;
	    total_entropy_num[k] = 0;
    }

    fprintf(compare_out, "[\n");
    // print
    for (int x = 0; x < num_debug_node; x++){
	if(x != 0)
		fprintf(compare_out, ",\n");
	bool compare_flag = false;
        IdKlDict msg_kl_dict;
        int cur_node_id = kl_dict.nodes[x]->node_id;
	int neighbor_label_type = 0; // 0 -> all neighbors are labelled, 1 -> all neighbors are unlabelled, 2 -> mixed
        // fprintf(ana_out, "==============================================================================================\n");
        // fprintf(ana_out, "Node %d's prior: %s:%f -- %s:%f   ground truth: %s   predict: %s   KL divergence: %f\n", cur_node_id,
        //        train_data->label_dict.GetKeyWithId(0).c_str(), factor_graph->var_node[cur_node_id].state_factor[0], train_data->label_dict.GetKeyWithId(1).c_str(), factor_graph->var_node[cur_node_id].state_factor[1],
        //       train_data->label_dict.GetKeyWithId(sample->node[cur_node_id]->label).c_str(), train_data->label_dict.GetKeyWithId(inf_label[cur_node_id]).c_str(),
        //       kl_dict.nodes[x]->kl_value);

        // message sorting
        // calculate the KL divergence
        // EQUATION 1
        for (int i = 0; i < factor_graph->var_node[cur_node_id].neighbor.size(); i++){
            double cur_kl = 0.0;
            for (int k = 0; k < num_label; k++){
                msg[k] = factor_graph->var_node[cur_node_id].belief[i][k];
                post[k] = label_prob[k][cur_node_id];
            }
	    if(msg[0] == 0){
		msg[0] += MINI_CHANGE; msg[1] -= MINI_CHANGE; 
	    }else if(msg[1] == 0){
		msg[1] += MINI_CHANGE; msg[0] -= MINI_CHANGE;
	    }
            BI_KL_DIV(msg, post, num_label, cur_kl);
//            msg_kl_dict.node_id.push_back(i);  // the index in this node
//            msg_kl_dict.id_kl_map.insert(make_pair(i, cur_kl));
            msg_kl_dict.nodes.push_back(new NodeKL(i, cur_kl));

	    if (conf->is_priorkl_bigger_than_msgkl==1 && kl_dict.nodes[x]->kl_value > cur_kl)
                compare_flag = true;
            if (conf->is_priorkl_bigger_than_msgkl==0 && kl_dict.nodes[x]->kl_value < cur_kl)
                compare_flag = true;
        }

	if(conf->is_priorkl_bigger_than_msgkl!=-1 && !compare_flag)
            continue;
	
	// TODO improve
//		double* entropy = new double[factor_graph->var_node[cur_node_id].neighbor.size() + 1];
	double * each_type_entropy = new double[num_edge_type + 1];
	int * each_type_entropy_num = new int[num_edge_type + 1];
	// memset(each_type_entropy, 0, num_edge_type + 1);
	// memset(each_type_entropy_num, 0, num_edge_type + 1);
	for(int k = 0; k < num_edge_type+1; k ++){
		each_type_entropy[k] = 0;
		each_type_entropy_num[k] = 0;
	}
	double cur_entropy = 0.0;
	ENTROPY(factor_graph->var_node[cur_node_id].state_factor, 2, cur_entropy);
	each_type_entropy[0] += cur_entropy;
	each_type_entropy_num[0] += 1;
	total_entropy[0] += cur_entropy;
	total_entropy_num[0] += 1;

	fprintf(ana_out, "==============================================================================================\n");
        fprintf(ana_out, "Node %d's prior: %s:%f -- %s:%f   ground truth: %s   predict: %s   KL divergence: %f   prior entropy: %f   post: %s:%f -- %s:%f\n", cur_node_id,
                train_data->label_dict.GetKeyWithId(0).c_str(), factor_graph->var_node[cur_node_id].state_factor[0],
                train_data->label_dict.GetKeyWithId(1).c_str(), factor_graph->var_node[cur_node_id].state_factor[1],
                train_data->label_dict.GetKeyWithId(sample->node[cur_node_id]->label).c_str(),
                train_data->label_dict.GetKeyWithId(inf_label[cur_node_id]).c_str(),
                kl_dict.nodes[x]->kl_value, cur_entropy, 
		train_data->label_dict.GetKeyWithId(0).c_str(), label_prob[0][cur_node_id],
                train_data->label_dict.GetKeyWithId(1).c_str(), label_prob[1][cur_node_id]);
	// TODO compare eauation and real
	CalcLinearBPEquation(sample, factor_graph, cur_node_id, label_prob, edge_factor, edge_num, compare_out, graph_out);
	if(x == num_debug_node - 1)
		fprintf(compare_out, "\n]");	
        // sort
        msg_kl_dict.sort_id_by_value(true);
        // print
        fprintf(ana_out, "++ SORT msg neighbor->node (KL(m_{i->x}||b_x) + KL(b_x||m_{i->x}))++\n");
        for (int i = 0; i < msg_kl_dict.nodes.size(); i++){
            int index = msg_kl_dict.nodes[i]->node_id;
            int edge_type = ((FactorNode*)factor_graph->var_node[cur_node_id].neighbor[index])->edge_type;
            int neighbor_id = (factor_graph->var_node[cur_node_id].neighbor[index]->neighbor[0])->id;
            neighbor_id = (neighbor_id == factor_graph->var_node[cur_node_id].id)
                      ? (factor_graph->var_node[cur_node_id].neighbor[index]->neighbor[1])->id : neighbor_id;
	    int x_label = sample->node[cur_node_id]->label;
            if (factor_graph->var_node[cur_node_id].belief[index][x_label] > factor_graph->var_node[cur_node_id].belief[index][1 - x_label])
                 continue;
	
	    // printf("before:\n");
	    // for(int k = 0; k < num_edge_type + 1; k++){
            //         if(k == 0) continue;
            //         printf("%s %f %d\n",train_data->edge_type_dict.GetKeyWithId(k-1).c_str(), each_type_entropy[k], each_type_entropy_num[k]);
            // }

	    // TODO improve
	    ENTROPY(factor_graph->var_node[cur_node_id].belief[index], 2, cur_entropy);
	    // TODO notice!!! edge_type is not the edge type
	    each_type_entropy[edge_type + 1] += cur_entropy;
	    each_type_entropy_num[edge_type + 1] += 1;
	    total_entropy[edge_type + 1] += cur_entropy;
	    total_entropy_num[edge_type + 1] += 1;
	    
	    // printf("after:\n");
	    // for(int k = 0; k < num_edge_type + 1; k++){
	    //        if(k == 0) continue;
	    //        printf("%s %f %d\n",train_data->edge_type_dict.GetKeyWithId(k-1).c_str(), each_type_entropy[k], each_type_entropy_num[k]);
	    // }
	    // fflush(stdout);

            fprintf(ana_out, "      Neighbor %d's prior: %s:%f -- %s:%f   msg: %s:%f -- %s:%f   edge type: %s   KL divergence: %f   ground truth: %s   predict: %s   msg entropy: %f\n", neighbor_id,
		    train_data->label_dict.GetKeyWithId(0).c_str(), factor_graph->var_node[neighbor_id].state_factor[0], train_data->label_dict.GetKeyWithId(1).c_str(), factor_graph->var_node[neighbor_id].state_factor[1],
                    train_data->label_dict.GetKeyWithId(0).c_str(), factor_graph->var_node[cur_node_id].belief[index][0], train_data->label_dict.GetKeyWithId(1).c_str(), factor_graph->var_node[cur_node_id].belief[index][1],
                    train_data->edge_type_dict.GetKeyWithId(edge_type).c_str(),
                    msg_kl_dict.nodes[i]->kl_value, train_data->label_dict.GetKeyWithId(sample->node[neighbor_id]->label).c_str(),
		    train_data->label_dict.GetKeyWithId(inf_label[neighbor_id]).c_str(), cur_entropy);
	    // TODO improve the analysis data
	    if(factor_graph->var_node[neighbor_id].label_type == Enum::KNOWN_LABEL){
		    if (i == 0){
				neighbor_label_type = Enum::KNOWN_LABEL;
		    }else if (neighbor_label_type == 1) {
				neighbor_label_type = 2;
		    }
	    }else{
		    if (i == 0){
				neighbor_label_type = Enum::UNKNOWN_LABEL;
	       	    } else if (neighbor_label_type == 0) {
				neighbor_label_type = 2;
		    }
	    }
        }

	// TODO improve the analysis data
	if (neighbor_label_type == 0)
		fprintf(ana_out, "NEIGHBORS SITUATION: ALL LABELED\n");
	else if (neighbor_label_type == 1)
		fprintf(ana_out, "NEIGHBORS SITUATION: ALL UNLABELED\n");
	else if (neighbor_label_type == 2)
		fprintf(ana_out, "NEIGHBORS SITUATION: MIXED\n");

	// normalize
	// TODO improve
	fprintf(ana_out, "For Entropy:\n");
	IdKlDict index_entropy_dict;
	for(int i = 0; i < num_edge_type + 1; i++){
		if(each_type_entropy_num[i] == 0)
			continue;
		each_type_entropy[i] /= each_type_entropy_num[i];
		index_entropy_dict.nodes.push_back(new NodeKL(i, each_type_entropy[i]));
	}

	//for(int k = 0; k < num_edge_type + 1; k++){
        //            if(k == 0) continue;
        //            printf("%s %f %d\n",train_data->edge_type_dict.GetKeyWithId(k-1).c_str(), each_type_entropy[k], each_type_entropy_num[k]);
        //    }

	index_entropy_dict.sort_id_by_value(true);  // ascending order
	int temp_flag = 0;
	for(int i = 0; i < index_entropy_dict.nodes.size(); i++){
		if(temp_flag != 0)
			fprintf(ana_out, " < ");
		else
			temp_flag = 1;
		if(index_entropy_dict.nodes[i]->node_id == 0){
			fprintf(ana_out, "prior: %f", index_entropy_dict.nodes[i]->kl_value);  // actually entropy
		}else{
			if(each_type_entropy_num[index_entropy_dict.nodes[i]->node_id] == 0)
				continue;
			fprintf(ana_out, "%s: %f", train_data->edge_type_dict.GetKeyWithId(index_entropy_dict.nodes[i]->node_id - 1).c_str(),
				   index_entropy_dict.nodes[i]->kl_value);
		}
	}
	fprintf(ana_out, "\n");

        // EQUATION 2
        // calculate the KL divergence
        IdKlDict msg_kl2_dict;
        for (int i = 0; i < factor_graph->var_node[cur_node_id].neighbor.size(); i++){
            double cur_kl = 0.0;
            double sum = 0.0;
	    int neighbor_id = (factor_graph->var_node[cur_node_id].neighbor[i]->neighbor[0])->id;
            neighbor_id = (neighbor_id == factor_graph->var_node[cur_node_id].id)
                          ? (factor_graph->var_node[cur_node_id].neighbor[i]->neighbor[1])->id : neighbor_id;
            for (int k = 0; k < num_label; k++)
                sum += factor_graph->var_node[neighbor_id].state_factor[k];
            for (int k = 0; k < num_label; k++){
                msg[k] = factor_graph->var_node[cur_node_id].belief[i][k];
                prior[k] = factor_graph->var_node[neighbor_id].state_factor[k]/sum;
            }
	    if(msg[0] == 0){
                msg[0] += MINI_CHANGE; msg[1] -= MINI_CHANGE;
            }else if(msg[1] == 0){
                msg[1] += MINI_CHANGE; msg[0] -= MINI_CHANGE;
            }
	    if(prior[0] == 0){
                prior[0] += MINI_CHANGE; prior[1] -= MINI_CHANGE;
            }else if(prior[1] == 0){
                prior[1] += MINI_CHANGE; prior[0] -= MINI_CHANGE;
            }
            BI_KL_DIV(prior, msg, num_label, cur_kl);
//	    printf("%f %f %f %f %f\n", prior[0], prior[1], msg[0], msg[1], cur_kl);
	    fflush(stdout);
//            msg_kl2_dict.node_id.push_back(i); // the index in this node
//            msg_kl2_dict.id_kl_map.insert(make_pair(i, cur_kl));
            msg_kl2_dict.nodes.push_back(new NodeKL(i, cur_kl));
        }
        // sort
        msg_kl2_dict.sort_id_by_value(true); // ????

        // print
        fprintf(ana_out, "++ SORT prior_i (KL(m_{i->x}||p_i) + KL(p_i||m_{i->x})) ++\n");
        for (int i = 0; i < msg_kl2_dict.nodes.size(); i++){
            int index = msg_kl2_dict.nodes[i]->node_id;
            int edge_type = ((FactorNode*)factor_graph->var_node[cur_node_id].neighbor[index])->edge_type;
            int neighbor_id = (factor_graph->var_node[cur_node_id].neighbor[index]->neighbor[0])->id;
            neighbor_id = (neighbor_id == factor_graph->var_node[cur_node_id].id)
                          ? (factor_graph->var_node[cur_node_id].neighbor[index]->neighbor[1])->id : neighbor_id;
	    int x_label = sample->node[cur_node_id]->label;
            if (factor_graph->var_node[cur_node_id].belief[index][x_label] > factor_graph->var_node[cur_node_id].belief[index][1 - x_label])
                continue;
	    // TODO find the index of ~msg
	    int in_index = -1;
	    for(int k = 0; k < factor_graph->var_node[neighbor_id].neighbor.size(); k ++){
	    	if(factor_graph->var_node[neighbor_id].neighbor[k]->neighbor[0]->id + factor_graph->var_node[neighbor_id].neighbor[k]->neighbor[1]->id == neighbor_id + cur_node_id)
	    		in_index = k;
	    }


	    // TODO improve, change the position
	    fprintf(ana_out, "      Neighbor %d's msg: %s:%f -- %s:%f   edge type: %s   KL divergence: %f   ground truth: %s   predict: %s   post: %s:%f -- %s:%f   ~msg:  %s:%f -- %s:%f\n", neighbor_id,
                    train_data->label_dict.GetKeyWithId(0).c_str(), factor_graph->var_node[cur_node_id].belief[index][0], train_data->label_dict.GetKeyWithId(1).c_str(), factor_graph->var_node[cur_node_id].belief[index][1],
                    train_data->edge_type_dict.GetKeyWithId(edge_type).c_str(),
                    msg_kl2_dict.nodes[i]->kl_value, train_data->label_dict.GetKeyWithId(sample->node[neighbor_id]->label).c_str(),
                    train_data->label_dict.GetKeyWithId(inf_label[neighbor_id]).c_str(),
		    train_data->label_dict.GetKeyWithId(0).c_str(), label_prob[0][neighbor_id],
                    train_data->label_dict.GetKeyWithId(1).c_str(), label_prob[1][neighbor_id], 
		    train_data->label_dict.GetKeyWithId(0).c_str(), factor_graph->var_node[neighbor_id].belief[in_index][0],
		    train_data->label_dict.GetKeyWithId(1).c_str(), factor_graph->var_node[neighbor_id].belief[in_index][1]);

	    double temp_dia = lambda[GetEdgeParameterId(edge_type, 0, 0)] + lambda[GetEdgeParameterId(edge_type, 1, 1)];
	    double temp_ndia = lambda[GetEdgeParameterId(edge_type, 0, 1)] + lambda[GetEdgeParameterId(edge_type, 1, 0)];
	    fprintf(ana_out, "      IT IS %15s, THE MATRIX |%6f|%6f|\n", temp_dia > temp_ndia ? "Homophily":"Anti-homophily",
				exp(lambda[GetEdgeParameterId(edge_type, 0, 0)]), exp(lambda[GetEdgeParameterId(edge_type, 0, 1)]));
	    fprintf(ana_out, "            %15s             |%6f|%6f|\n", " ",
				exp(lambda[GetEdgeParameterId(edge_type, 1, 0)]), exp(lambda[GetEdgeParameterId(edge_type, 1, 1)]));
	    
	    // TODO improve the analysis data
	    if (factor_graph->var_node[neighbor_id].label_type == Enum::KNOWN_LABEL)
		    continue;

            // calculate the KL divergence
            IdKlDict msg_kl3_dict;
            for (int j = 0; j < factor_graph->var_node[neighbor_id].neighbor.size(); j++){
                double cur_kl = 0.0;
                for (int k = 0; k < num_label; k++){
                    msg[k] = factor_graph->var_node[cur_node_id].belief[i][k];
                    sub_msg[k] = factor_graph->var_node[neighbor_id].belief[j][k];
                }
		if(msg[0] == 0){
             	   msg[0] += MINI_CHANGE; msg[1] -= MINI_CHANGE;
            	}else if(msg[1] == 0){
                   msg[1] += MINI_CHANGE; msg[0] -= MINI_CHANGE;
            	}
		if(sub_msg[0] == 0){
		   sub_msg[0] += MINI_CHANGE; sub_msg[1] -= MINI_CHANGE;
		}else if(sub_msg[1] == 0){
		   sub_msg[1] += MINI_CHANGE; sub_msg[0] -= MINI_CHANGE;
		}
                BI_KL_DIV(msg, sub_msg, num_label, cur_kl);
//                msg_kl3_dict.node_id.push_back(j); // the index in the node
//                msg_kl3_dict.id_kl_map.insert(make_pair(j, cur_kl));
                msg_kl3_dict.nodes.push_back(new NodeKL(j, cur_kl));
            }
            // sort
            msg_kl3_dict.sort_id_by_value(temp_dia > temp_ndia); // temp_dia > temp_ndia? ascend: descend
            // print
            for (int j = 0; j < msg_kl3_dict.nodes.size(); j++){
                int nn_index = msg_kl3_dict.nodes[j]->node_id;
                int nn_edge_type = ((FactorNode*)factor_graph->var_node[neighbor_id].neighbor[nn_index])->edge_type;
                int nn_neighbor_id = (factor_graph->var_node[neighbor_id].neighbor[nn_index]->neighbor[0])->id;
                nn_neighbor_id = (nn_neighbor_id == factor_graph->var_node[neighbor_id].id)
                              ? (factor_graph->var_node[neighbor_id].neighbor[nn_index]->neighbor[1])->id : nn_neighbor_id;
		if (cur_node_id == nn_neighbor_id)
			continue;
                fprintf(ana_out, "            2-hop Neighbor %d's msg: %s:%f -- %s:%f   edge type: %s   KL divergence: %f   ground truth: %s   predict: %s\n",
                        nn_neighbor_id, train_data->label_dict.GetKeyWithId(0).c_str(), factor_graph->var_node[neighbor_id].belief[nn_index][0], train_data->label_dict.GetKeyWithId(1).c_str(), factor_graph->var_node[neighbor_id].belief[nn_index][1],
                        train_data->edge_type_dict.GetKeyWithId(nn_edge_type).c_str(), msg_kl3_dict.nodes[j]->kl_value, 
			train_data->label_dict.GetKeyWithId(sample->node[nn_neighbor_id]->label).c_str(),
			train_data->label_dict.GetKeyWithId(inf_label[nn_neighbor_id]).c_str());
            }
        }
	msg_kl_dict.clear();
    }

    	// TODO improve
    fprintf(ana_out, "For Entropy:\n");
    IdKlDict entropy_dict;
    for(int i = 0; i < num_edge_type + 1; i++){
	if(total_entropy_num[i] == 0)
		continue;
	total_entropy[i] /= total_entropy_num[i];
	entropy_dict.nodes.push_back(new NodeKL(i, total_entropy[i]));
     }
     entropy_dict.sort_id_by_value(true);  // ascending order
     for(int i = 0; i < entropy_dict.nodes.size(); i++){
	if(i != 0)
		fprintf(ana_out, " < ");
	if(entropy_dict.nodes[i]->node_id == 0){
		fprintf(ana_out, "prior: %f", entropy_dict.nodes[i]->kl_value);  // actually entropy
	}else{
		fprintf(ana_out, "%s: %f", train_data->edge_type_dict.GetKeyWithId(entropy_dict.nodes[i]->node_id - 1).c_str(),
				entropy_dict.nodes[i]->kl_value);
	}
     }
     fprintf(ana_out, "\n");

     int is_first = 1;
     char temp_for_now[1024];
     string edge_effect_factor_str = "";
     for(int e = 0; e < num_edge_type; e ++){
           if (is_first == 1){
                  edge_factor[e] /= (double)(edge_num[e]);
                  sprintf(temp_for_now, "%s:%f", train_data->edge_type_dict.GetKeyWithId(e).c_str(), edge_factor[e]);
                  edge_effect_factor_str += temp_for_now;
                  is_first = 0;
           }else{
		  edge_factor[e] /= (double)(edge_num[e]);
                  sprintf(temp_for_now, ", %s:%f", train_data->edge_type_dict.GetKeyWithId(e).c_str(), edge_factor[e]);
                  edge_effect_factor_str += temp_for_now;

           }
     }
     printf("\n\n%s\n\n", edge_effect_factor_str.c_str());
     fflush(stdout);
}

void CRFModel::CalcLinearBPEquation(DataSample* sample, FactorGraph* factor_graph, int node_id, double** label_prob, double* global_edge_factor, int* global_edge_num, FILE * compare_out, FILE * graph_out)
{
	double linear_post[num_label];
	int is_edge_type[num_edge_type] = {0};
	// each edge's 00 and 01 and then prior0, calc post0, true post itmes0
	double items[num_edge_type * 2 + 3] = {0.0};
	int label0_id = train_data->label_dict.GetIdConst("0");
        items[num_edge_type * 2] = factor_graph->var_node[node_id].state_factor[label0_id];
	items[num_edge_type * 2 + 2] = label_prob[label0_id][node_id];
	int neg_effect = 0, pos_effect = 0;
	int neg_edge_effect[num_edge_type] = {0}, pos_edge_effect[num_edge_type] = {0};
	double neg_effect_value[num_edge_type] = {0.0}, pos_effect_value[num_edge_type] = {0.0};
	double effect_factor[num_edge_type] = {0.0}; // self-designed sign for wrong direction
	double pos_neighbor_effect_value = 0, neg_neighbor_effect_value = 0;
	int node_label = sample->node[node_id]->label;

	for(int i = 0; i < num_label; i ++){
		double cur_post = factor_graph->var_node[node_id].state_factor[i];
		for(int xj = 0; xj < factor_graph->var_node[node_id].neighbor.size(); xj++){

			int neighbor_id = (factor_graph->var_node[node_id].neighbor[xj]->neighbor[0])->id;
            		neighbor_id = (neighbor_id == factor_graph->var_node[node_id].id)? (factor_graph->var_node[node_id].neighbor[xj]->neighbor[1])->id : neighbor_id;
			int in_index = factor_graph->var_node[neighbor_id].neighbor_pos[node_id];
			int edge_type = ((FactorNode*)factor_graph->var_node[node_id].neighbor[xj])->edge_type;
			if (i == label0_id)
				is_edge_type[edge_type] = 1;
			double neighbor_effect = 0.0;
			for(int j = 0; j < num_label; j++){
				neighbor_effect += (0.5 * (exp(lambda[GetEdgeParameterId(edge_type, i, j)]) - exp(lambda[GetEdgeParameterId(edge_type, 1 - i, j)])) * (label_prob[j][neighbor_id] - factor_graph->var_node[neighbor_id].belief[in_index][j]));
				cur_post += (0.5 * (exp(lambda[GetEdgeParameterId(edge_type, i, j)]) - exp(lambda[GetEdgeParameterId(edge_type, 1 - i, j)])) * (label_prob[j][neighbor_id] - factor_graph->var_node[neighbor_id].belief[in_index][j]));
				if(i == label0_id){
					if (i == j){
						items[edge_type * 2] += 0.5 * exp(lambda[GetEdgeParameterId(edge_type, i, j)]) * (label_prob[j][neighbor_id] - factor_graph->var_node[neighbor_id].belief[in_index][j]);
						items[edge_type * 2 + 1] -= 0.5 * exp(lambda[GetEdgeParameterId(edge_type, 1 - i, j)]) * (label_prob[j][neighbor_id] - factor_graph->var_node[neighbor_id].belief[in_index][j]);
					}else if (i != j){
						items[edge_type * 2 + 1] += 0.5 * exp(lambda[GetEdgeParameterId(edge_type, i, j)]) * (label_prob[j][neighbor_id] - factor_graph->var_node[neighbor_id].belief[in_index][j]); 
						items[edge_type * 2] -= 0.5 * exp(lambda[GetEdgeParameterId(edge_type, 1 - i, j)]) * (label_prob[j][neighbor_id] - factor_graph->var_node[neighbor_id].belief[in_index][j]);
					}
				}
			}
			if (i == label0_id){
				if (neighbor_effect < 0){
					neg_neighbor_effect_value += neighbor_effect;
					neg_effect += 1;
					neg_edge_effect[edge_type] += 1;
					neg_effect_value[edge_type] += neighbor_effect;

					if(node_label == label0_id){ // wrong direction
						int neigh_label = sample->node[neighbor_id]->label;
						if (label_prob[neigh_label][neighbor_id] - factor_graph->var_node[neighbor_id].belief[in_index][neigh_label] > label_prob[1 - neigh_label][neighbor_id] - factor_graph->var_node[neighbor_id].belief[in_index][1 - neigh_label]){ // this neighbor's classification is right
							effect_factor[edge_type] -= neighbor_effect;
							global_edge_factor[edge_type] -= neighbor_effect;
						}
					}else{ // right direction
						// wrong neighbor shoule lead to wrong effect
						int neigh_label = sample->node[neighbor_id]->label;
                                                if (label_prob[neigh_label][neighbor_id] - factor_graph->var_node[neighbor_id].belief[in_index][neigh_label] < label_prob[1 - neigh_label][neighbor_id] - factor_graph->var_node[neighbor_id].belief[in_index][1 - neigh_label]){ // this neighbor's classification is wrong
                                                        effect_factor[edge_type] -= neighbor_effect;
							global_edge_factor[edge_type] -= neighbor_effect;
                                                }
					}
				}else{
					pos_neighbor_effect_value += neighbor_effect;
					pos_effect += 1;
					pos_edge_effect[edge_type] += 1;
					pos_effect_value[edge_type] += neighbor_effect;

					if(node_label != label0_id){ // wrong direction
						int neigh_label = sample->node[neighbor_id]->label;
                                                if (label_prob[neigh_label][neighbor_id] - factor_graph->var_node[neighbor_id].belief[in_index][neigh_label] > label_prob[1 - neigh_label][neighbor_id] - factor_graph->var_node[neighbor_id].belief[in_index][1 - neigh_label]){ // this neighbor's classification is right
                                                        effect_factor[edge_type] += neighbor_effect;
							global_edge_factor[edge_type] += neighbor_effect;
                                                }
					}else{ // right direction
						// wrong neighbor shoule lead to wrong effect
                                                int neigh_label = sample->node[neighbor_id]->label;
                                                if (label_prob[neigh_label][neighbor_id] - factor_graph->var_node[neighbor_id].belief[in_index][neigh_label] < label_prob[1 - neigh_label][neighbor_id] - factor_graph->var_node[neighbor_id].belief[in_index][1 - neigh_label]){ // this neighbor's classification is wrong
                                                        effect_factor[edge_type] += neighbor_effect;
							global_edge_factor[edge_type] += neighbor_effect;
                                                }
					}
				}
			}
		}
		linear_post[i] = cur_post;
	}

	items[num_edge_type * 2 + 1] = linear_post[label0_id];
	double sum = 0.0;
	for(int i = 0; i <= num_edge_type * 2; i ++)
		sum += items[i];
	fprintf(graph_out, "prior:%f ",items[num_edge_type * 2]);
	fprintf(graph_out, "calc_post:%f ", items[num_edge_type * 2 + 1]);
	fprintf(graph_out, "true_post:%f ", items[num_edge_type * 2 + 2]);
	for(int i = 0; i < num_edge_type; i++){
		if(is_edge_type[i] != 1) continue;
		fprintf(graph_out, "%s:%f ", train_data->edge_type_dict.GetKeyWithId(i).c_str(), items[i * 2] + items[i * 2 + 1]);
	}
	fprintf(graph_out, "\n");


	// fprintf(compare_out, "Node %d's prior: %s:%f %s:%f   post: %s:%f %s:%f   calc post: %s:%f %s:%f   num neighbors: %d   neg neigh(label 0): %d   pos neigh(label 0): %d   pos neigh effect(label 0): %f   neg neigh effect(label 0): %f\n", node_id,
	// 	        train_data->label_dict.GetKeyWithId(0).c_str(), factor_graph->var_node[node_id].state_factor[0], train_data->label_dict.GetKeyWithId(1).c_str(), factor_graph->var_node[node_id].state_factor[1],	
	// 		train_data->label_dict.GetKeyWithId(0).c_str(), label_prob[0][node_id], train_data->label_dict.GetKeyWithId(1).c_str(), label_prob[1][node_id],
	// 		train_data->label_dict.GetKeyWithId(0).c_str(), linear_post[0], train_data->label_dict.GetKeyWithId(1).c_str(), linear_post[1],
	// 		factor_graph->var_node[node_id].neighbor.size(), neg_effect, pos_effect, pos_neighbor_effect_value, neg_neighbor_effect_value);
	// fprintf(compare_out, "    neg neigh:%d = ", neg_effect);
	// int is_first = 1;
	// for(int e = 0; e < num_edge_type; e ++){
	// 	if (neg_edge_effect[e] != 0 && is_first == 1){
	// 		fprintf(compare_out, "%s:%d ", train_data->edge_type_dict.GetKeyWithId(e).c_str(), neg_edge_effect[e]);
	// 		is_first = 0;
	// 	}else if (neg_edge_effect[e] != 0){
	// 		fprintf(compare_out, "+ %s:%d ", train_data->edge_type_dict.GetKeyWithId(e).c_str(), neg_edge_effect[e]);
	// 	}
	// }
	// fprintf(compare_out, "\n");
	// fprintf(compare_out, "    pos neigh:%d = ", pos_effect);
	// is_first = 1;
	// for(int e = 0; e < num_edge_type; e ++){
        //         if (pos_edge_effect[e] != 0 && is_first == 1){
        //                 fprintf(compare_out, "%s:%d ", train_data->edge_type_dict.GetKeyWithId(e).c_str(), pos_edge_effect[e]);
        //                 is_first = 0;
        //         }else if (pos_edge_effect[e] != 0){
        //                 fprintf(compare_out, "+ %s:%d ", train_data->edge_type_dict.GetKeyWithId(e).c_str(), pos_edge_effect[e]);
        //         }
        // }
	// fprintf(compare_out, "\n");

	string pos_str = "", neg_str = "", pos_effect_str = "", neg_effect_str = "", edge_effect_factor_str="";
	char temp[1024];
	int is_first = 1;
        for(int e = 0; e < num_edge_type; e ++){
                if (pos_edge_effect[e] != 0 && is_first == 1){
			sprintf(temp, ", \"%s\":%d", train_data->edge_type_dict.GetKeyWithId(e).c_str(), pos_edge_effect[e]);
			pos_str += temp;
			sprintf(temp, ", \"%s\":%f", train_data->edge_type_dict.GetKeyWithId(e).c_str(), pos_effect_value[e]);
			pos_effect_str += temp;
                        is_first = 0;
                }else if (pos_edge_effect[e] != 0){
                        sprintf(temp, ", \"%s\":%d", train_data->edge_type_dict.GetKeyWithId(e).c_str(), pos_edge_effect[e]);
			pos_str += temp;
			sprintf(temp, ", \"%s\":%f", train_data->edge_type_dict.GetKeyWithId(e).c_str(), pos_effect_value[e]);
                        pos_effect_str += temp;
                }
        }
	is_first = 1;
	for(int e = 0; e < num_edge_type; e ++){
                if (neg_edge_effect[e] != 0 && is_first == 1){
                        sprintf(temp, ", \"%s\":%d", train_data->edge_type_dict.GetKeyWithId(e).c_str(), neg_edge_effect[e]);
			neg_str += temp;
			sprintf(temp, ", \"%s\":%f", train_data->edge_type_dict.GetKeyWithId(e).c_str(), neg_effect_value[e]);
                        neg_effect_str += temp;
                        is_first = 0;
                }else if (neg_edge_effect[e] != 0){
                        sprintf(temp, ", \"%s\":%d", train_data->edge_type_dict.GetKeyWithId(e).c_str(), neg_edge_effect[e]);
			neg_str += temp;
			sprintf(temp, ", \"%s\":%f", train_data->edge_type_dict.GetKeyWithId(e).c_str(), neg_effect_value[e]);
                        neg_effect_str += temp;
                }
        }
	is_first = 1;
	for(int e = 0; e < num_edge_type; e ++){
		global_edge_num[e] += (pos_edge_effect[e] + neg_edge_effect[e]);
		if (pos_edge_effect[e] + neg_edge_effect[e] != 0 && is_first == 1){
			effect_factor[e] /= (double)(pos_edge_effect[e] + neg_edge_effect[e]);
			sprintf(temp, "\"%s\":%f", train_data->edge_type_dict.GetKeyWithId(e).c_str(), effect_factor[e]);
			edge_effect_factor_str += temp;
			is_first = 0;
		}else if(pos_edge_effect[e] + neg_edge_effect[e] != 0){
			effect_factor[e] /= (double)(pos_edge_effect[e] + neg_edge_effect[e]);
                        sprintf(temp, ", \"%s\":%f", train_data->edge_type_dict.GetKeyWithId(e).c_str(), effect_factor[e]);
                        edge_effect_factor_str += temp;
		}
	}




	fprintf(compare_out, "{\"Node\":%d,\n \"prior\":{\"%s\":%f, \"%s\":%f},\n \"post\":{\"%s\":%f, \"%s\":%f},\n \"calc_post\":{\"%s\":%f, \"%s\":%f},\n \"neighbor\":{\"total\":%d,\n\t \"pos_label0\":{\"num\":%d%s},\n\t \"neg_label0\":{\"num\":%d%s},\n\t \"pos_effect_label0\":{\"value\":%f%s},\n\t \"neg_effect_label0\":{\"value\":%f%s},\n\t \"edge_factor\":{%s}}}", node_id,
			train_data->label_dict.GetKeyWithId(0).c_str(), factor_graph->var_node[node_id].state_factor[0], train_data->label_dict.GetKeyWithId(1).c_str(), factor_graph->var_node[node_id].state_factor[1],
			train_data->label_dict.GetKeyWithId(0).c_str(), label_prob[0][node_id], train_data->label_dict.GetKeyWithId(1).c_str(), label_prob[1][node_id],
			train_data->label_dict.GetKeyWithId(0).c_str(), linear_post[0], train_data->label_dict.GetKeyWithId(1).c_str(), linear_post[1],
			factor_graph->var_node[node_id].neighbor.size(), pos_effect, pos_str.c_str(), neg_effect, neg_str.c_str(), pos_neighbor_effect_value, pos_effect_str.c_str(), neg_neighbor_effect_value, neg_effect_str.c_str(), edge_effect_factor_str.c_str());

}

int CRFModel::FindAnchorNode(DataSample *sample, FactorGraph *factor_graph, double ** label_prob){
    int n = sample->num_node;
    int count = 0;
    for(int i = 0; i < n; i++){
        VariableNode cur_node = factor_graph->var_node[i];
        int cur_node_label = sample->node[i]->label;
        if(sample->node[i]->label_type == Enum::KNOWN_LABEL){
            count ++;
            continue;
        }
        if(label_prob[cur_node_label] < label_prob[1 - cur_node_label]){
            continue;
        }

        double cur_post =  factor_graph->var_node[i].state_factor[cur_node_label];
        int neighbor_len = (int)cur_node.neighbor.size();
        for(int j = 0; j < neighbor_len; j++){
            FactorNode* cur_neighbor_factor = (FactorNode*)cur_node.neighbor[j];
            VariableNode* cur_neighbor_var = (VariableNode*)cur_neighbor_factor->neighbor[1 - cur_neighbor_factor->neighbor_pos[cur_node.id]];
            int edge_type = cur_neighbor_factor->edge_type;
            if(sample->node[cur_neighbor_var->id]->label_type == Enum::KNOWN_LABEL){
                int in_index = cur_neighbor_var->neighbor_pos[cur_neighbor_factor->id];
                cur_post += (0.5 * (exp(lambda[GetEdgeParameterId(edge_type, cur_node_label, 0)]) - exp(lambda[GetEdgeParameterId(edge_type, 1 - cur_node_label, 0)])) * (label_prob[0][cur_neighbor_var->id] - factor_graph->var_node[cur_neighbor_var->id].belief[in_index][0]));
                cur_post += (0.5 * (exp(lambda[GetEdgeParameterId(edge_type, cur_node_label, 1)]) - exp(lambda[GetEdgeParameterId(edge_type, 1 - cur_node_label, 1)])) * (label_prob[1][cur_neighbor_var->id] - factor_graph->var_node[cur_neighbor_var->id].belief[in_index][1]));
            }else if (cur_neighbor_factor->func->GetValue(0, 0) > cur_neighbor_factor->func->GetValue(0, 1)){
                int in_index = cur_neighbor_var->neighbor_pos[cur_neighbor_factor->id];
		double temp_value = 0.0;
		if(sample->node[cur_neighbor_var->id]->label == cur_node_label)
			temp_value = 0.5;
                cur_post += (0.5 * (exp(lambda[GetEdgeParameterId(edge_type, cur_node_label, cur_node_label)]) - exp(lambda[GetEdgeParameterId(edge_type, 1- cur_node_label , cur_node_label)])) * (temp_value - factor_graph->var_node[cur_neighbor_var->id].belief[in_index][cur_node_label]));
                cur_post += (0.5 * (exp(lambda[GetEdgeParameterId(edge_type, cur_node_label, 1- cur_node_label)]) - exp(lambda[GetEdgeParameterId(edge_type, 1-cur_node_label, 1-cur_node_label)])) * (1.0 - temp_value - factor_graph->var_node[cur_neighbor_var->id].belief[in_index][1 - cur_node_label]));
            }else{
                int in_index = cur_neighbor_var->neighbor_pos[cur_neighbor_factor->id];
		double temp_value = 1.0;
		if(sample->node[cur_neighbor_var->id]->label != cur_node_label)
			temp_value = 0.5;
                cur_post += (0.5 * (exp(lambda[GetEdgeParameterId(edge_type, cur_node_label, cur_node_label)]) - exp(lambda[GetEdgeParameterId(edge_type, 1- cur_node_label , cur_node_label)])) * (temp_value - factor_graph->var_node[cur_neighbor_var->id].belief[in_index][cur_node_label]));
                cur_post += (0.5 * (exp(lambda[GetEdgeParameterId(edge_type, cur_node_label, 1- cur_node_label)]) - exp(lambda[GetEdgeParameterId(edge_type, 1-cur_node_label, 1-cur_node_label)])) * (1.0 - temp_value - factor_graph->var_node[cur_neighbor_var->id].belief[in_index][1 - cur_node_label]));
            }
        }
        if(cur_post > 0.5) count ++;
    }
    return count;
}


int CRFModel::FindAnchorNodeV1(DataSample *sample, FactorGraph *factor_graph, double **label_prob) {
    int n = sample->num_node;
    for(int i = 0; i < n; i++){
        if(sample->node[i]->label_type == Enum::KNOWN_LABEL){
            anchor_nodes.insert(i);
        }
    }
    int ne = factor_graph->num_node;
    for(int bfsi = 0; bfsi < ne; bfsi++){
        Node* cur_node = factor_graph->bfs_node[bfsi];
        if(cur_node->id < n){ // variable node
            VariableNode* cur_var_node = (VariableNode*)cur_node;
            int cur_var_label = sample->node[cur_var_node->id]->label;
	    if(sample->node[cur_var_node->id]->label_type == Enum::KNOWN_LABEL || label_prob[cur_var_label][cur_var_node->id] < label_prob[1 - cur_var_label][cur_var_node->id]){
		    continue;
	    }
            double cur_post = cur_var_node->state_factor[cur_var_label];
            int cur_neighbor_len = (int)cur_var_node->neighbor.size();
            for(int j = 0; j < cur_neighbor_len; j++){
                FactorNode* cur_neighbor_factor = (FactorNode*)cur_var_node->neighbor[j];
                VariableNode* cur_neighbor_var = (VariableNode*)cur_neighbor_factor->neighbor[1 - cur_neighbor_factor->neighbor_pos[cur_var_node->id]];
                int edge_type = cur_neighbor_factor->edge_type;

                if(anchor_nodes.find(cur_neighbor_var->id) == anchor_nodes.end()){ // not find
                    if(cur_neighbor_factor->func->GetValue(0, 0) > cur_neighbor_factor->func->GetValue(0, 1)){
                        double temp_value = 0.0;
                        if(sample->node[cur_neighbor_var->id]->label == cur_var_label){
                            temp_value = 0.5;
                        } else{
                            temp_value = 0.0;
                        }
                        int in_index = cur_neighbor_var->neighbor_pos[cur_neighbor_factor->id];
                        cur_post += (0.5 * (exp(lambda[GetEdgeParameterId(edge_type, cur_var_label, cur_var_label)]) - exp(lambda[GetEdgeParameterId(edge_type, 1- cur_var_label , cur_var_label)])) * (temp_value - factor_graph->var_node[cur_neighbor_var->id].belief[in_index][cur_var_label]));
                        cur_post += (0.5 * (exp(lambda[GetEdgeParameterId(edge_type, cur_var_label, 1- cur_var_label)]) - exp(lambda[GetEdgeParameterId(edge_type, 1- cur_var_label, 1-cur_var_label)])) * (1.0 - temp_value - factor_graph->var_node[cur_neighbor_var->id].belief[in_index][1 - cur_var_label]));
                    }else{
                        double temp_value = 0.0;
                        if(sample->node[cur_neighbor_var->id]->label != cur_var_label){
                            temp_value = 0.5;
                        } else{
                            temp_value = 1.0;
                        }
                        int in_index = cur_neighbor_var->neighbor_pos[cur_neighbor_factor->id];
                        cur_post += (0.5 * (exp(lambda[GetEdgeParameterId(edge_type, cur_var_label, cur_var_label)]) - exp(lambda[GetEdgeParameterId(edge_type, 1- cur_var_label , cur_var_label)])) * (temp_value - factor_graph->var_node[cur_neighbor_var->id].belief[in_index][cur_var_label]));
                        cur_post += (0.5 * (exp(lambda[GetEdgeParameterId(edge_type, cur_var_label, 1- cur_var_label)]) - exp(lambda[GetEdgeParameterId(edge_type, 1- cur_var_label, 1-cur_var_label)])) * (1.0 - temp_value - factor_graph->var_node[cur_neighbor_var->id].belief[in_index][1 - cur_var_label]));
                    }
                }else{
                    int in_index = cur_neighbor_var->neighbor_pos[cur_neighbor_factor->id];
                    cur_post += (0.5 * (exp(lambda[GetEdgeParameterId(edge_type, cur_var_label, 0)]) - exp(lambda[GetEdgeParameterId(edge_type, 1 - cur_var_label, 0)])) * (label_prob[0][cur_neighbor_var->id] - factor_graph->var_node[cur_neighbor_var->id].belief[in_index][0]));
                    cur_post += (0.5 * (exp(lambda[GetEdgeParameterId(edge_type, cur_var_label, 1)]) - exp(lambda[GetEdgeParameterId(edge_type, 1 - cur_var_label, 1)])) * (label_prob[1][cur_neighbor_var->id] - factor_graph->var_node[cur_neighbor_var->id].belief[in_index][1]));
                }
            }

            if(cur_post > 0.5){
                anchor_nodes.insert(cur_var_node->id);
            }
        }
    }
    return (int)anchor_nodes.size();
}


int CRFModel::FindAnchorNodeV2(DataSample *sample, FactorGraph *factor_graph, double **label_prob) {
    int n = sample->num_node;
    for(int i = 0; i < n; i++){
        if(sample->node[i]->label_type == Enum::KNOWN_LABEL){
            anchor_nodes.insert(i);
        }
    }
    int ne = factor_graph->num_node;
    for(int bfsi = 0; bfsi < ne; bfsi++){
        Node* cur_node = factor_graph->bfs_node[bfsi];
        if(cur_node->id < n){ // variable node
            VariableNode* cur_var_node = (VariableNode*)cur_node;
            int cur_var_label = sample->node[cur_var_node->id]->label;
	    if(sample->node[cur_var_node->id]->label_type == Enum::KNOWN_LABEL || label_prob[cur_var_label][cur_var_node->id] < label_prob[1 - cur_var_label][cur_var_node->id]){
                    continue;
            }
            double cur_post = cur_var_node->state_factor[cur_var_label];
            int cur_neighbor_len = (int)cur_var_node->neighbor.size();
            for(int j = 0; j < cur_neighbor_len; j++){
                FactorNode* cur_neighbor_factor = (FactorNode*)cur_var_node->neighbor[j];
                VariableNode* cur_neighbor_var = (VariableNode*)cur_neighbor_factor->neighbor[1 - cur_neighbor_factor->neighbor_pos[cur_var_node->id]];
                int edge_type = cur_neighbor_factor->edge_type;

                if(anchor_nodes.find(cur_neighbor_var->id) == anchor_nodes.end()){ // not find
                    double temp_value = 0.5;
                    int in_index = cur_neighbor_var->neighbor_pos[cur_neighbor_factor->id];
                    cur_post += (0.5 * (exp(lambda[GetEdgeParameterId(edge_type, cur_var_label, cur_var_label)]) - exp(lambda[GetEdgeParameterId(edge_type, 1- cur_var_label , cur_var_label)])) * (temp_value - factor_graph->var_node[cur_neighbor_var->id].belief[in_index][cur_var_label]));
                    cur_post += (0.5 * (exp(lambda[GetEdgeParameterId(edge_type, cur_var_label, 1- cur_var_label)]) - exp(lambda[GetEdgeParameterId(edge_type, 1- cur_var_label, 1-cur_var_label)])) * (1.0 - temp_value - factor_graph->var_node[cur_neighbor_var->id].belief[in_index][1 - cur_var_label]));
                }else{
                    int in_index = cur_neighbor_var->neighbor_pos[cur_neighbor_factor->id];
                    cur_post += (0.5 * (exp(lambda[GetEdgeParameterId(edge_type, cur_var_label, 0)]) - exp(lambda[GetEdgeParameterId(edge_type, 1 - cur_var_label, 0)])) * (label_prob[0][cur_neighbor_var->id] - factor_graph->var_node[cur_neighbor_var->id].belief[in_index][0]));
                    cur_post += (0.5 * (exp(lambda[GetEdgeParameterId(edge_type, cur_var_label, 1)]) - exp(lambda[GetEdgeParameterId(edge_type, 1 - cur_var_label, 1)])) * (label_prob[1][cur_neighbor_var->id] - factor_graph->var_node[cur_neighbor_var->id].belief[in_index][1]));
                }
            }

            if(cur_post > 0.5){
                anchor_nodes.insert(cur_var_node->id);
            }
        }
    }
    return (int)anchor_nodes.size();
}


int CRFModel::FindAnchorNodeBound(DataSample *sample, FactorGraph *factor_graph){
    int n = sample->num_node;
    for(int i = 0; i < n; i++){
        double b0_lower = factor_graph->var_node[i].GetBliefBound(0, 0); // label 0 lower
        double b1_lower = factor_graph->var_node[i].GetBliefBound(1, 0); // label 0 lower
        if(b0_lower > 0.5){
            if(sample->node[i]->label == 0){
                // it is anchor nod
		anchor_nodes.insert(i);
            }
        }else if(b1_lower > 0.5){
            if(sample->node[i]->label == 1){
                // it is anchor node
		anchor_nodes.insert(i);
            }
        }
    }
    return (int)anchor_nodes.size();
}


// TODO analyse
double CRFModel::CalcPartialLabeledGradientForSample(DataSample* sample, FactorGraph* factor_graph, double* gradient, bool is_write_analyse)
{   
	int spec_node = -1;
	int n = sample->num_node;
	int m = sample->num_edge;
	
	//****************************************************************
	// Belief Propagation 1: labeled data are given.
	//****************************************************************

	factor_graph->labeled_given = true;
	factor_graph->ClearDataForSumProduct();
	
	// Set state_factor
	for (int i = 0; i < n; i++)
	{
		double sum = 0.0;
		double* p_lambda = lambda;
		for (int y = 0; y < num_label; y++)
		{
			if (sample->node[i]->label_type == Enum::KNOWN_LABEL && y != sample->node[i]->label)
			{
				factor_graph->SetVariableStateFactor(i, y, 0);
				sum += 0.0;
			}
			else
			{
				double v = 1;
				for (int t = 0; t < sample->node[i]->num_attrib; t++)
					v *= exp(p_lambda[sample->node[i]->attrib[t]] * sample->node[i]->value[t]);                
				factor_graph->SetVariableStateFactor(i, y, v);
				sum += v;
			}
			p_lambda += num_attrib_type;
		}
		// TODO change the base
        	for (int y = 0; y < num_label; y++){
           		 factor_graph->SetVariableStateFactor(i, y, factor_graph->var_node[i].state_factor[y]/sum);
        	}
		if(i == spec_node){
                        factor_graph->SetVariableStateFactor(i, 0, factor_graph->var_node[i].state_factor[0] - 0.001);
                        factor_graph->SetVariableStateFactor(i, 1, factor_graph->var_node[i].state_factor[1] + 0.001);
                }
	}
	
	factor_graph->BeliefPropagation(conf->max_infer_iter);
	factor_graph->CalculateMarginal();    

	/***
	* Gradient = E_{Y|Y_L} f_i - E_{Y} f_i
	*/

	// calc gradient part : + E_{Y|Y_L} f_i
	for (int i = 0; i < n; i++)
	{
		for (int y = 0; y < num_label; y++)
		{
			for (int t = 0; t < sample->node[i]->num_attrib; t++)
				gradient[GetAttribParameterId(y, sample->node[i]->attrib[t])] += sample->node[i]->value[t] * factor_graph->var_node[i].marginal[y];
		}
	}

	for (int i = 0; i < m; i++)
	{
		for (int a = 0; a < num_label; a++)
			for (int b = 0; b < num_label; b++)
			{
				gradient[GetEdgeParameterId(sample->edge[i]->edge_type, a, b)] += factor_graph->factor_node[i].marginal[a][b];
			}
    }

	//****************************************************************
	// Belief Propagation 2: labeled data are not given.
	//****************************************************************

	factor_graph->ClearDataForSumProduct();
	factor_graph->labeled_given = false;

	for (int i = 0; i < n; i++)
	{
		double sum = 0.0;
		double* p_lambda = lambda;
		for (int y = 0; y < num_label; y++)
		{
			double v = 1;
			for (int t = 0; t < sample->node[i]->num_attrib; t++)
				v *= exp(p_lambda[sample->node[i]->attrib[t]] * sample->node[i]->value[t]);
			factor_graph->SetVariableStateFactor(i, y, v);
			p_lambda += num_attrib_type;
			sum += v;
		}
		// TODO change the base
        	for (int y = 0; y < num_label; y++){
            		factor_graph->SetVariableStateFactor(i, y, factor_graph->var_node[i].state_factor[y]/sum);
        	}
		if(i == spec_node){
                        factor_graph->SetVariableStateFactor(i, 0, factor_graph->var_node[i].state_factor[0] - 0.001);
                        factor_graph->SetVariableStateFactor(i, 1, factor_graph->var_node[i].state_factor[1] + 0.001);
                }
	}    

	factor_graph->BeliefPropagation(conf->max_infer_iter);
	factor_graph->CalculateMarginal();
	
	// calc gradient part : - E_{Y} f_i
	for (int i = 0; i < n; i++)
	{
		for (int y = 0; y < num_label; y++)
		{
			for (int t = 0; t < sample->node[i]->num_attrib; t++)
				gradient[GetAttribParameterId(y, sample->node[i]->attrib[t])] -= sample->node[i]->value[t] * factor_graph->var_node[i].marginal[y];
		}
	}
	for (int i = 0; i < m; i++)
	{
		for (int a = 0; a < num_label; a++)
			for (int b = 0; b < num_label; b++)
			{
				gradient[GetEdgeParameterId(sample->edge[i]->edge_type, a, b)] -= factor_graph->factor_node[i].marginal[a][b];
			}
	}
	
	// Calculate gradient & log-likelihood
	double f = 0.0, Z = 0.0;

	// \sum \lambda_i * f_i
	for (int i = 0; i < n; i++)
	{
		int y = sample->node[i]->label;
		for (int t = 0; t < sample->node[i]->num_attrib; t++)
			f += lambda[this->GetAttribParameterId(y, sample->node[i]->attrib[t])] * sample->node[i]->value[t];
		if(isnan(f) == true)
			printf("there is a nan\n");
	}
	for (int i = 0; i < m; i++)
	{
		int a = sample->node[sample->edge[i]->a]->label;
		int b = sample->node[sample->edge[i]->b]->label;        
		f += lambda[this->GetEdgeParameterId(sample->edge[i]->edge_type, a, b)];
		if(isnan(f) == true)
			printf("there is a nan\n");
	}

	// calc log-likelihood
	//  using Bethe Approximation
	for (int i = 0; i < n; i++)
	{
		for (int y = 0; y < num_label; y++)
		{
			for (int t = 0; t < sample->node[i]->num_attrib; t++){
				Z += lambda[this->GetAttribParameterId(y, sample->node[i]->attrib[t])] * sample->node[i]->value[t] * factor_graph->var_node[i].marginal[y];
				if(isnan(Z) == true)
					printf("there is a nan\n");
			}
		}
	}
	for (int i = 0; i < m; i++)
	{
		for (int a = 0; a < num_label; a++)
			for (int b = 0; b < num_label; b++)
			{
				Z += lambda[this->GetEdgeParameterId(sample->edge[i]->edge_type, a, b)] * factor_graph->factor_node[i].marginal[a][b];
				if(isnan(Z) == true)
					printf("there is a nan\n");
			}
	}
	// Edge entropy
	for (int i = 0; i < m; i++)
	{
		double h_e = 0.0;
		for (int a = 0; a < num_label; a++)
			for (int b = 0; b < num_label; b++)
			{
				if (factor_graph->factor_node[i].marginal[a][b] > 1e-10)
					h_e += - factor_graph->factor_node[i].marginal[a][b] * log(factor_graph->factor_node[i].marginal[a][b]);
				if(isnan(h_e) == true)
					printf("there is a nan\n");
			}
		Z += h_e;
	}
	// Node entroy
	for (int i = 0; i < n; i++)
	{
		double h_v = 0.0;
		for (int a = 0; a < num_label; a++){
			if (fabs(factor_graph->var_node[i].marginal[a]) > 1e-10)
				h_v += - factor_graph->var_node[i].marginal[a] * log(factor_graph->var_node[i].marginal[a]);
			if(isnan(h_v) == true)
				printf("there is a nan\n");
		}
		Z -= h_v * ((int)factor_graph->var_node[i].neighbor.size() - 1);
	}
	
	f -= Z;
	if (isnan(f) == true)
		printf("there is a nan");
	
	// Let's take a look of current accuracy

	factor_graph->ClearDataForMaxSum();
	factor_graph->labeled_given = true;

	for (int i = 0; i < n; i++)
	{
		double* p_lambda = lambda;
		double sum = 0.0;
		for (int y = 0; y < num_label; y++)
		{
			// TODO known nodes' state factor is their label
            		if (sample->node[i]->label_type == Enum::KNOWN_LABEL && y != sample->node[i]->label) {
                		factor_graph->SetVariableStateFactor(i, y, 0);
				sum += 0.0;
            		}else{
                		double v = 1.0;
				for (int t = 0; t < sample->node[i]->num_attrib; t++)
					v *= exp(p_lambda[sample->node[i]->attrib[t]] * sample->node[i]->value[t]);
				factor_graph->SetVariableStateFactor(i, y, v);
				sum += v;
			}
			p_lambda += num_attrib_type;
		}
		// TODO change the base
                for (int y = 0; y < num_label; y++){
                        factor_graph->SetVariableStateFactor(i, y, factor_graph->var_node[i].state_factor[y]/sum);
                }
		if(i == spec_node){
                        factor_graph->SetVariableStateFactor(i, 0, factor_graph->var_node[i].state_factor[0] - 0.001);
                        factor_graph->SetVariableStateFactor(i, 1, factor_graph->var_node[i].state_factor[1] + 0.001);
                }
	}    

	// factor_graph->MaxSumPropagation(conf->max_infer_iter);
	factor_graph->BeliefPropagation(conf->max_infer_iter);

	int* inf_label = new int[n];
	// TODO analyse
	int* inf_label_state = new int[n];

	double** label_prob = new double*[num_label];
	for (int p = 0; p < num_label; p++)
		label_prob[p] = new double[n];

	for (int i = 0; i < n; i++)
	{
		// TODO analyse
        	int stateybest = -1;
        	double statevbest = 0, statev = 0;

		int ybest = -1;
		double vbest, v;
		double vsum = 0.0;
		for (int y = 0; y < num_label; y++)
		{
			v = factor_graph->var_node[i].state_factor[y];
			// TODO analyse
			statev = factor_graph->var_node[i].state_factor[y];

			for (int t = 0; t < factor_graph->var_node[i].neighbor.size(); t++){
				v *= factor_graph->var_node[i].belief[t][y];
			}
			if (ybest < 0 || v > vbest)
				ybest = y, vbest = v;
			// TODO analyse
            		if (stateybest < 0 || statev > statevbest)
                		stateybest = y, statevbest = statev;

			label_prob[y][i] = v;
			vsum += v;
		}

		inf_label[i] = ybest;
		// TODO analyse
		inf_label_state[i] = stateybest;

		for (int y = 0; y < num_label; y++){
			label_prob[y][i] /= vsum;
		}
	}

	int hit = 0, miss = 0;
	int hitu = 0, missu = 0;
	int hitv = 0, missv = 0;

	int cnt[10][10];
	int ucnt[10][10];
	int vcnt[10][10];

	memset(ucnt, 0, sizeof(ucnt));
	memset(cnt, 0, sizeof(cnt));
	memset(vcnt, 0, sizeof(vcnt));

	FILE *pred_out = fopen((conf->out_dir + "/pred.txt").c_str(), "w");
	// TODO analyse
    //FILE *analyse_out = fopen((conf->out_dir + "/analyse.txt").c_str(), "a");
    	// TODO compare equation and reality
//	FILE * compare_out = fopen((conf->out_dir + "/compare.txt").c_str(), "a");
//	FILE * graph_out = fopen((conf->out_dir + "/graph.txt").c_str(), "a");

	for (int i = 0; i < n; i++)
	{
		fprintf(pred_out, "%s\n", train_data->label_dict.GetKeyWithId(inf_label[i]).c_str());
		if (inf_label[i] == sample->node[i]->label){
			hit++;
            // TODO server
//			if (is_write_analyse)
//                		AnalyseLogic(sample, factor_graph, inf_label_state, inf_label, i, analyse_out);
		}else{
			miss++;
            // TODO server
//			if (is_write_analyse)
//                		AnalyseLogic(sample, factor_graph, inf_label_state, inf_label, i, analyse_out);
		}
		cnt[inf_label[i]][sample->node[i]->label]++;

		if (sample->node[i]->label_type == Enum::UNKNOWN_LABEL)
		{
			if (sample->node[i]->type1 == Enum1::TEST){
				if (inf_label[i] == sample->node[i]->label)
					hitu++;
				else
					missu++;
				ucnt[inf_label[i]][sample->node[i]->label]++;
			}else if (sample->node[i]->type1 == Enum1::VALID){
				if (inf_label[i] == sample->node[i]->label)
					hitv ++;
				else
					missv ++;
				vcnt[inf_label[i]][sample->node[i]->label]++;
			}
			// ucnt[inf_label[i]][sample->node[i]->label]++;
		}
	}
    // TODO serve
  //  if (is_write_analyse){
  //  	AnalyseLogic(sample, factor_graph, inf_label, label_prob, analyse_out, compare_out, graph_out);
  //  }

	// TODO analyse
    //fflush(analyse_out);
    //fclose(analyse_out);
    //fflush(compare_out);
    //fclose(compare_out);
	fclose(pred_out);
	
	// int number = FindAnchorNodeV2(sample, factor_graph, label_prob);
	// printf("anchor nodes: %d\n", number);
	// fflush(stdout);

	int dat[12];
	memset(dat, 0, sizeof(dat));

	hit += dat[0]; hitu += dat[1];
	miss += dat[2]; missu += dat[3];
	cnt[0][0] += dat[4]; cnt[0][1] += dat[5]; cnt[1][0] += dat[6]; cnt[1][1] += dat[7];
	ucnt[0][0] += dat[8]; ucnt[0][1] += dat[9]; ucnt[1][0] += dat[10]; ucnt[1][1] += dat[11];

	printf("A_HIT  = %4d, U_HIT  = %4d, V_HIT  = %4d\n", hit, hitu, hitv);
	printf("A_MISS = %4d, U_MISS = %4d, V_MISS = %4d\n", miss, missu, missv);

	//!!!!!!!! make sure, the first instance is "positive"

	// 0 -> positive
	// 1 -> negative

	double ap = (double)cnt[0][0] / (cnt[0][0] + cnt[0][1]);
	double up = (double)ucnt[0][0] / (ucnt[0][0] + ucnt[0][1]);
	double vp = (double)vcnt[0][0] / (vcnt[0][0] + vcnt[0][1]);

	double ar = (double)cnt[0][0] / (cnt[0][0] + cnt[1][0]);
	double ur = (double)ucnt[0][0] / (ucnt[0][0] + ucnt[1][0]);
	double vr = (double)vcnt[0][0] / (vcnt[0][0] + vcnt[1][0]);

	double af = 2 * ap * ar / (ap + ar);
	double uf = 2 * up * ur / (up + ur);
	double vf = 2 * vp * vr / (vp + vr);

	printf("A_Accuracy  = %.4lf     U_Accuracy  = %.4lf     V_Accuracy  = %.4lf\n", (double)hit / (hit + miss), (double)hitu / (hitu + missu), (double)hitv / (hitv + missv));
	printf("A_Precision = %.4lf     U_Precision = %.4lf     V_Precision = %.4lf\n", ap, up, vp);
	printf("A_Recall    = %.4lf     U_Recall    = %.4lf     V_Recall    = %.4lf\n", ar, ur, vr);
	printf("A_F1        = %.4lf     U_F1        = %.4lf     V_F1        = %.4lf\n", af, uf, vf);
		
	fflush(stdout);

	FILE* fprob = fopen((conf->out_dir + "/uncertainty.txt").c_str(), "w");
	for (int i = 0; i < n; i++)
	{
		if (sample->node[i]->label_type == Enum::KNOWN_LABEL)
		{
			for (int y = 0; y < num_label; y++)
				fprintf(fprob, "%s -1 ", train_data->label_dict.GetKeyWithId(y).c_str());
			fprintf(fprob, "\n");
		}
		else
		{
			for (int y = 0; y < num_label; y++)
				fprintf(fprob, "%s %.4lf ", train_data->label_dict.GetKeyWithId(y).c_str(), label_prob[y][i]);
			fprintf(fprob, "\n");
		}
	}
	fclose(fprob);
    
	factor_graph->ClearDataForBoundPropagation();
	factor_graph->BoundPropagation(conf->max_infer_iter);
	int num_anchor_nodes = FindAnchorNodeBound(sample, factor_graph);
	printf("num anchor nodes: %d\n", num_anchor_nodes);
	fflush(stdout);

	delete[] inf_label;
	for (int y = 0; y < num_label; y++)
		delete[] label_prob[y];
	delete[] label_prob;

	return f;
}

void CRFModel::SelfEvaluate()
{
	int ns = train_data->num_sample;
	int tot, hit;

	tot = hit = 0;
	for (int s = 0; s < ns; s++)
	{
		DataSample* sample = train_data->sample[s];
		FactorGraph* factor_graph = &sample_factor_graph[s];
		
		int n = sample->num_node;
		int m = sample->num_edge;
		
		factor_graph->InitGraph(n, m, num_label);
		// Add edge info
		for (int i = 0; i < m; i++)
		{
            // TODO server
			factor_graph->AddEdge(sample->edge[i]->a, sample->edge[i]->b, func_list[sample->edge[i]->edge_type], sample->edge[i]->edge_type);
		}        
		factor_graph->GenPropagateOrder();

		factor_graph->ClearDataForMaxSum();

		for (int i = 0; i < n; i++)
		{
			double* p_lambda = lambda;
			double sum = 0.0;
			for (int y = 0; y < num_label; y++)
			{
				double v = 1.0;
				for (int t = 0; t < sample->node[i]->num_attrib; t++)
					v *= exp(p_lambda[sample->node[i]->attrib[t]] * sample->node[i]->value[t]);
				factor_graph->SetVariableStateFactor(i, y, v);
				sum += v;
				p_lambda += num_attrib_type;
			}
			// TODO change the base
                	for (int y = 0; y < num_label; y++){
                        	factor_graph->SetVariableStateFactor(i, y, factor_graph->var_node[i].state_factor[y]/sum);
                	}
		}    

		factor_graph->MaxSumPropagation(conf->max_infer_iter);

		int* inf_label = new int[n];
		for (int i = 0; i < n; i++)
		{
			int ybest = -1;
			double vbest, v;

			for (int y = 0; y < num_label; y++)
			{
				v = factor_graph->var_node[i].state_factor[y];
				for (int t = 0; t < factor_graph->var_node[i].neighbor.size(); t++)
					v *= factor_graph->var_node[i].belief[t][y];
				if (ybest < 0 || v > vbest)
					ybest = y, vbest = v;
			}

			inf_label[i] = ybest;
		}

		int curt_tot, curt_hit;
		curt_tot = curt_hit = 0;
		for (int i = 0; i < n; i++)
		{   
			curt_tot ++;
			if (inf_label[i] == sample->node[i]->label) curt_hit++;
		}
		
		printf("Accuracy %4d / %4d : %.6lf\n", curt_hit, curt_tot, (double)curt_hit / curt_tot);
		hit += curt_hit;
		tot += curt_tot;

		delete[] inf_label;
	}

	printf("Overall Accuracy %4d / %4d : %.6lf\n", hit, tot, (double)hit / tot);
}

void CRFModel::InitEvaluate(Config* conf, DataSet* test_data)
{
	this->conf = conf;
	this->test_data = test_data;
}

void CRFModel::Evalute()
{
	int ns = test_data->num_sample;
	int tot, hit;

    tot = hit = 0;

	FILE* fout = fopen((conf->out_dir + "/" +  conf->pred_file).c_str(), "w");

	for (int s = 0; s < ns; s++)
	{
		DataSample* sample = test_data->sample[s];
		FactorGraph* factor_graph = new FactorGraph();
		
		int n = sample->num_node;
		int m = sample->num_edge;
		
		factor_graph->InitGraph(n, m, num_label);
		// Add edge info
		for (int i = 0; i < m; i++)
		{
            // TODO server
			factor_graph->AddEdge(sample->edge[i]->a, sample->edge[i]->b, func_list[sample->edge[i]->edge_type], sample->edge[i]->edge_type);
		}        
		factor_graph->GenPropagateOrder();

		factor_graph->ClearDataForMaxSum();

		for (int i = 0; i < n; i++)
		{
			double* p_lambda = lambda;
			double sum = 0.0;
			for (int y = 0; y < num_label; y++)
			{
				double v = 1.0;
				for (int t = 0; t < sample->node[i]->num_attrib; t++)
					v *= exp(p_lambda[sample->node[i]->attrib[t]] * sample->node[i]->value[t]);
				factor_graph->SetVariableStateFactor(i, y, v);
				sum += v;
				p_lambda += num_attrib_type;
			}
			// TODO change the base
                	for (int y = 0; y < num_label; y++){
                        	factor_graph->SetVariableStateFactor(i, y, factor_graph->var_node[i].state_factor[y]/sum);
                	}
		}    

		factor_graph->MaxSumPropagation(conf->max_infer_iter);

		int* inf_label = new int[n];
		for (int i = 0; i < n; i++)
		{
			int ybest = -1;
			double vbest, v;

			for (int y = 0; y < num_label; y++)
			{
				v = factor_graph->var_node[i].state_factor[y];
				for (int t = 0; t < factor_graph->var_node[i].neighbor.size(); t++)
					v *= factor_graph->var_node[i].belief[t][y];
				if (ybest < 0 || v > vbest)
					ybest = y, vbest = v;
			}

			inf_label[i] = ybest;
		}

		int curt_tot, curt_hit;
		curt_tot = curt_hit = 0;
		for (int i = 0; i < n; i++)
		{   
			curt_tot ++;
			if (inf_label[i] == sample->node[i]->label) curt_hit++;
		}
		
		printf("Accuracy %4d / %4d : %.6lf\n", curt_hit, curt_tot, (double)curt_hit / curt_tot);
		hit += curt_hit;
		tot += curt_tot;

		// to zz: just print inf_labe[0]
		for (int i = 0; i < n; i++)
		{
			fprintf(fout, "%s\n", train_data->label_dict.GetKeyWithId(inf_label[i]).c_str());
		}

		delete[] inf_label;
    }

	printf("Overall Accuracy %4d / %4d : %.6lf\n", hit, tot, (double)hit / tot);
	fclose(fout);
}

void CRFModel::Clean()
{
	if (lambda) delete[] lambda;
	if (sample_factor_graph) delete[] sample_factor_graph;

	for (int i = 0; i < num_edge_type; i++)
		delete func_list[i];
	delete[] func_list;
}

void CRFModel::LoadModel(const char* filename)
{
	FILE* fin = fopen(filename, "r");
	char buf[MAX_BUF_SIZE];
	vector<string> tokens;
	for (;;)
	{
		if (fgets(buf, MAX_BUF_SIZE, fin) == NULL)
			break;
		tokens = CommonUtil::StringTokenize(buf);
		if (tokens[0] == "#node")
		{
			int class_id = train_data->label_dict.GetIdConst(tokens[1]);
			int feat_id = train_data->attrib_dict.GetIdConst(tokens[2]);
			int pid = GetAttribParameterId(class_id, feat_id);
			double value = atof(tokens[3].c_str());
			lambda[pid] = value;
		}
		if (tokens[0] == "#edge")
		{
			int type_id = train_data->edge_type_dict.GetIdConst(tokens[1]);
			int id1 = train_data->label_dict.GetIdConst(tokens[2]);
			int id2 = train_data->label_dict.GetIdConst(tokens[3]);
			double value = atof(tokens[4].c_str());
			int i = GetEdgeParameterId(type_id, id1, id2);
			lambda[i] = value;
		}
	}
	fclose(fin);
	printf("Load %s finished.\n", filename);
}

void CRFModel::SaveModel(const char* filename)
{
	FILE* fout = fopen((conf->out_dir + "/" + filename).c_str(), "w");
	for (int i = 0; i < num_label; i++)
	{
		string cl = train_data->label_dict.GetKeyWithId(i);
		for (int j = 0; j < num_attrib_type; j++)
		{
			string feature = train_data->attrib_dict.GetKeyWithId(j);
			int pid = GetAttribParameterId(i, j);
			fprintf(fout, "#node %s %s %f\n", cl.c_str(), feature.c_str(), lambda[pid]);
		}
	}
	for (int T = 0; T < num_edge_type; T++)
	{
		string c0 = train_data->edge_type_dict.GetKeyWithId(T);
		for (int i = 0; i < num_label; i++)
		{
			string c1 = train_data->label_dict.GetKeyWithId(i);
			for (int j = i; j < num_label; j++)
			{
				string c2 = train_data->label_dict.GetKeyWithId(j);
				int pid = GetEdgeParameterId(T, i, j);
				fprintf(fout, "#edge %s %s %s %f\n", c0.c_str(), c1.c_str(), c2.c_str(), lambda[pid]);
			}
		}
	}
	fclose(fout);
}

void CRFModel::Estimate(Config* conf)
{
	DataSet* dataset;

	dataset = new DataSet();
	dataset->LoadData(conf->train_file.c_str(), conf);
	dataset->label_dict.SaveMappingDict(conf->dict_file.c_str());

	printf("num_label = %d\n", dataset->num_label);
	printf("num_sample = %d\n", dataset->num_sample);
	printf("num_edge_type = %d\n", dataset->num_edge_type);
	printf("num_attrib_type = %d\n", dataset->num_attrib_type);
	
	InitTrain(conf, dataset);    

	printf("Start Training...\n");
	fflush(stdout);
	if (conf->method == "LBP") Train(conf->is_write_analyse);
	else if (conf->method == "MH")
	{
		if (conf->state_file != "") 
			LoadStateFile(conf->state_file.c_str());
		MHTrain(conf);
		SavePred("uncertainty.txt");
	}
	else if (conf->method == "MH1")
	{
		if (conf->state_file != "") 
			LoadStateFile(conf->state_file.c_str());
		MHTrain1(conf);
		SavePred("uncertainty.txt");
	}
	else
	{
		printf("Method error!\n");
		return;
	}
	
	// TODO 5 analyse
    	if(!conf->is_write_analyse) {
        	SaveModel(conf->dst_model_file.c_str());
    	}
}

void CRFModel::EstimateContinue(Config* conf)
{
	DataSet* dataset;

	dataset = new DataSet();
	dataset->LoadData(conf->train_file.c_str(), conf);
	dataset->label_dict.SaveMappingDict(conf->dict_file.c_str());

	printf("num_label = %d\n", dataset->num_label);
	printf("num_sample = %d\n", dataset->num_sample);
	printf("num_edge_type = %d\n", dataset->num_edge_type);
	printf("num_attrib_type = %d\n", dataset->num_attrib_type);
	
	InitTrain(conf, dataset);

	// TODO analyse
	LoadModel((conf->src_dir + conf->src_model_file).c_str());

	// TODO analyse
	if (conf->method == "LBP") Train(conf->is_write_analyse);
	else if (conf->method == "MH")
	{
		if (conf->state_file != "") 
			LoadStateFile(conf->state_file.c_str());
		MHTrain(conf);
		SavePred("uncertainty.txt");
	}
	else if (conf->method == "MH1")
	{
		if (conf->state_file != "") 
			LoadStateFile(conf->state_file.c_str());
		MHTrain1(conf);
		SavePred("uncertainty.txt");
	}
	else
	{
		printf("Method error!\n");
		return;
	}
	
	// TODO 5 analyse
    	// if(!conf->is_write_analyse) {
        // 	SaveModel(conf->dst_model_file.c_str());
    	// }
	SaveModel((conf->dst_model_file).c_str());
}

void CRFModel::Inference(Config* conf)
{
	DataSet* dataset;

	dataset = new DataSet();
	dataset->LoadData(conf->train_file.c_str(), conf);
	dataset->label_dict.SaveMappingDict(conf->dict_file.c_str());

	printf("num_label = %d\n", dataset->num_label);
	printf("num_sample = %d\n", dataset->num_sample);
	printf("num_edge_type = %d\n", dataset->num_edge_type);
	printf("num_attrib_type = %d\n", dataset->num_attrib_type);
	
	InitTrain(conf, dataset);
	// TODO analyse
	LoadModel((conf->src_dir + conf->src_model_file).c_str());
	
	if (conf->method == "LBP")
	{
		InitEvaluate(conf, dataset);
		Evalute();
	}
	else if ((conf->method == "MH") || (conf->method == "MH1"))
	{
		if (conf->state_file != "") 
			LoadStateFile(conf->state_file.c_str());
		MHEvaluate(conf->max_infer_iter, true);
	}
	else
	{
		printf("Method error!\n");
		return;
	}
}
