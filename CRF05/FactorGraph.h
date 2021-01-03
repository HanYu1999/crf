#pragma once

#include "Util.h"
#include <math.h>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
using std::vector;
using std::map;
using std::make_pair;
using std::set;
class FactorFunction
{
public:
	virtual double GetValue(int y1, int y2) = 0;
};

class Node
{
public:
	int             id;
	int             num_label;

	vector<Node*>   neighbor;
	vector<double*> belief;
	map<int, int>   neighbor_pos;

	double          *msg;
	vector<double *> msg_bound;

	virtual void Init(int num_label) = 0;
	void BasicInit(int num_label);
	void NormalizeMessage();

	void AddNeighbor(Node* ng)
	{
		neighbor_pos.insert(make_pair(ng->id, neighbor.size()));
		neighbor.push_back(ng);

		belief.push_back(MatrixUtil::GetDoubleArr(num_label));
		msg_bound.push_back(MatrixUtil::GetDoubleArr(num_label * 2));
	}

	virtual void BeliefPropagation(double* diff_max, bool labeled_given) = 0;
	virtual void MaxSumPropagation(double* diff_max, bool labeled_given) = 0;
	virtual void BoundPropagation() = 0;

	void SetMsgBound(int u, int label, int is_upper, double value){
   		int p = neighbor_pos[u];
		msg_bound[p][label * num_label + is_upper] = value;
    	}

	double GetMsgBound(int u, int label, int is_upper){
		int p = neighbor_pos[u];
		return msg_bound[p][label * num_label + is_upper];
	}

	void GetMessageFrom(int u, double* msgvec, double* diff_max)
	{
		int p = neighbor_pos[u];
		for (int y = 0; y < num_label; y++)
		{
			if (fabs(belief[p][y] - msgvec[y]) > *diff_max)
				*diff_max = fabs(belief[p][y] - msgvec[y]);
			belief[p][y] = msgvec[y];
		}
	}
	
	virtual ~Node()
	{
		for (int i = 0; i < belief.size(); i++)
			delete[] belief[i];
		if (msg)
			delete[] msg;
	}
};

class VariableNode : public Node
{
public:
	int     y;
	int     label_type;

	double* state_factor;

	double* marginal;

	double * prior_bound;
	double * belief_bound;

	VariableNode()
	{
		state_factor = NULL;
	}

	virtual void Init(int num_label);
	virtual void BeliefPropagation(double* diff_max, bool labeled_given);
	virtual void MaxSumPropagation(double* diff_max, bool labeled_given);
	virtual void BoundPropagation();

	virtual double GetPriorBound(int label, int is_upper);
	virtual double GetBliefBound(int label, int is_upper);
	virtual void SetPriorBound(int label, int is_upper, double value);
	virtual void SetBliefBound(int label, int is_upper, double value);

	virtual ~VariableNode()
	{
		if (state_factor) delete[] state_factor;
		if (marginal) delete[] marginal;
	}
};

class FactorNode : public Node
{
public:
	FactorFunction  *func;
	double **marginal;

	// TODO server
	int edge_type;

	virtual void Init(int num_label);
	virtual void BeliefPropagation(double* diff_max, bool labeled_given);
	virtual void MaxSumPropagation(double* diff_max, bool labeled_given);
	virtual void BoundPropagation();

	virtual ~FactorNode() 
	{
		if (marginal)
		{
			for (int i = 0; i < num_label; i++)
				if (marginal[i]) delete[] marginal[i];
			delete[] marginal;
		}
	}
};

class FactorGraph
{
public:    
	int                 n, m, num_label;
	int                 num_node;

	bool                converged;
	double              diff_max;

	bool                labeled_given;

	VariableNode*       var_node;
	FactorNode*         factor_node;
	Node**              p_node;
	Node**              bfs_node;

	// For each subgraph (connected component), we select one node as entry
	vector<Node*>       entry;

	set<int> anchor_nodes;

	int                 factor_node_used;

	void InitGraph(int n, int m, int num_label);
    // TODO server
	void AddEdge(int a, int b, FactorFunction* func, int edge_type);
	void GenPropagateOrder();
	void ClearDataForSumProduct();
	void ClearDataForMaxSum();
	void ClearDataForBoundPropagation();

	// int FindAnchorNode();

	void SetVariableLabel(int u, int y)
	{
		var_node[u].y = y;
	}
	void SetVariableStateFactor(int u, int y, double v)
	{
		var_node[u].state_factor[y] = v;
	}
	
	// Sum-Product
	void BeliefPropagation(int max_iter);
	void CalculateMarginal();

	// Max-Sum
	void MaxSumPropagation(int max_iter);
	
	void BoundPropagation(int max_iter);

	void Clean();
	~FactorGraph()
	{
		Clean();
	}
};
