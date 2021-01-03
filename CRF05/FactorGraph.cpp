#include "FactorGraph.h"
#include "Constant.h"

/**************************************************
 Node
**************************************************/

void Node::BasicInit(int num_label)
{
	this->num_label = num_label;
	msg = new double[num_label];
}

void VariableNode::Init(int num_label)
{
	BasicInit(num_label);

	state_factor = MatrixUtil::GetDoubleArr(num_label);

	marginal = MatrixUtil::GetDoubleArr(num_label);

	prior_bound = MatrixUtil::GetDoubleArr(num_label * 2);
	belief_bound = MatrixUtil::GetDoubleArr(num_label * 2);
}

void FactorNode::Init(int num_label)
{
	BasicInit(num_label);

	marginal = new double*[num_label];
	for (int i = 0; i < num_label; i++)
		marginal[i] = new double[num_label];
}

void Node::NormalizeMessage()
{
	double s = 0.0;
	for (int y = 0; y < num_label; y++)
		s += msg[y];
	if(s == 0.0)
		s += 1e-5;
	for (int y = 0; y < num_label; y++)
		msg[y] /= s;
}

void VariableNode::BeliefPropagation(double* diff_max, bool labeled_given)
{
	double product;

	for (int i = 0; i < neighbor.size(); i++)
	{
		FactorNode* f = (FactorNode*) neighbor[i];
		for (int y = 0; y < num_label; y++)
		{
			product = this->state_factor[y];
			for (int j = 0; j < neighbor.size(); j++)
				if (i != j)
					product *= this->belief[j][y];
			msg[y] = product;
		}

		NormalizeMessage();
		f->GetMessageFrom(id, msg, diff_max);
	}
}

void FactorNode::BeliefPropagation(double* diff_max, bool labeled_given)
{
	for (int i = 0; i < 2; i++)
	{
		// TODO labelled node's msgs to its neighbors is its labels
		// TODO labelled node ignore the message to it
		if (labeled_given && ((VariableNode*)neighbor[1 - i])->label_type == Enum::KNOWN_LABEL)
		{
			// for (int y = 0; y < num_label; y++)
			// 	msg[y] = 0;
			// msg[((VariableNode*)neighbor[i])->y] = 1.0;

			double temp[num_label];
                        for (int y = 0; y < num_label; y ++)
                                temp[y] = 0;
                        temp[((VariableNode*)neighbor[1 - i])->y] = 1.0;
                        for (int y = 0; y < num_label; y++){
                                double s = 0;
                                for (int y1 = 0; y1 < num_label; y1 ++)
                                	s += func->GetValue(y, y1) * temp[y1];
                                msg[y] = s;
                        }
                        NormalizeMessage();

		}
		else
		{
			for (int y = 0; y < num_label; y++)
			{
				double s = 0;
				for (int y1 = 0; y1 < num_label; y1++)
					s += func->GetValue(y, y1) * belief[1 - i][y1];
				msg[y] = s;
			}
			NormalizeMessage();
		}

		neighbor[i]->GetMessageFrom(id, msg, diff_max);
	}
}

void VariableNode::MaxSumPropagation(double* diff_max, bool labeled_given)
{
	double product;

	for (int i = 0; i < neighbor.size(); i++)
	{
		FactorNode* f = (FactorNode*) neighbor[i];
		for (int y = 0; y < num_label; y++)        
		{
			product = this->state_factor[y];
			for (int j = 0; j < neighbor.size(); j++)
				if (i != j)
					product *= this->belief[j][y];
			msg[y] = product;
		}
		NormalizeMessage();
		f->GetMessageFrom(id, msg, diff_max);
	}
}

void FactorNode::MaxSumPropagation(double* diff_max, bool labeled_given)
{
	for (int i = 0; i < 2; i++)    
	{
		// TODO labelled node's msgs to its neighbors is its labels
                // TODO labelled node ignore the message to it
		if (labeled_given && ((VariableNode*)neighbor[1 - i])->label_type == Enum::KNOWN_LABEL)
		{
			// for (int y = 0; y < num_label; y++)
			// 	msg[y] = 0;
			// msg[((VariableNode*)neighbor[i])->y] = 1.0;
			
			double temp[num_label];
			for (int y = 0; y < num_label; y ++)
				temp[y] = 0;
			temp[((VariableNode*)neighbor[1 - i])->y] = 1.0;
			for (int y = 0; y < num_label; y++){
				double s = func->GetValue(y, 0) * temp[0];
				double tmp;
				for (int y1 = 0; y1 < num_label; y1 ++){
					tmp = func->GetValue(y, y1) * temp[y1];
			 		if (tmp > s) s= tmp;
				}
				msg[y] = s;
			}
			NormalizeMessage();
		}
		else
		{
			for (int y = 0; y < num_label; y++)
			{
				double s = func->GetValue(y, 0) * belief[1 - i][0];
				double tmp;
				for (int y1 = 0; y1 < num_label; y1 ++)
				{
					tmp = func->GetValue(y, y1) * belief[1 - i][y1];
					if (tmp > s) s = tmp;
				}
				msg[y] = s;
			}
			NormalizeMessage();
		}

		neighbor[i]->GetMessageFrom(id, msg, diff_max);
	}
}

void VariableNode::BoundPropagation(){
	if(this->label_type == Enum::KNOWN_LABEL) return;
	double belief0_upper, belief0_lower, belief1_upper, belief1_lower;
	belief0_upper = this->GetPriorBound(0, 1);
	belief0_lower = this->GetPriorBound(0, 0);
	belief1_upper = this->GetPriorBound(1, 1);
	belief1_lower = this->GetPriorBound(1, 0);
	for(int i = 0; i < neighbor.size(); i++){
		FactorNode* f = (FactorNode*)neighbor[i];
		belief0_lower += this->GetMsgBound(f->id, 0, 0);
		belief1_lower += this->GetMsgBound(f->id, 1, 0);
		belief0_upper += this->GetMsgBound(f->id, 0, 1);
		belief1_upper += this->GetMsgBound(f->id, 1, 1);
	}
	if(belief0_lower > this->GetBliefBound(0, 0)) {
		this->SetBliefBound(0, 0, belief0_lower);
	}else{
		double prior0_lower = this->GetBliefBound(0, 0) - (belief0_lower - this->GetPriorBound(0, 0));
		if(prior0_lower > this->GetPriorBound(0, 0)) this->SetPriorBound(0, 0, prior0_lower);
	}
	if(belief1_lower > this->GetBliefBound(1, 0)) {
		this->SetBliefBound(1, 0, belief1_lower);
	}else{
		double prior1_lower = this->GetBliefBound(1, 0) - (belief1_lower - this->GetPriorBound(1, 0));
		if(prior1_lower > this->GetPriorBound(1, 0)) this->SetPriorBound(1, 0, prior1_lower);
	}
	if(belief0_upper < this->GetBliefBound(0, 1)) {
		this->SetBliefBound(0, 1, belief0_upper);
	}else{
		double prior0_upper = this->GetBliefBound(0, 1) - (belief0_upper - this->GetPriorBound(0, 1));
		if(prior0_upper < this->GetPriorBound(0, 1)) this->SetPriorBound(0, 1, prior0_upper);
	}
	if(belief1_upper < this->GetBliefBound(1, 1)) {
		this->SetBliefBound(1, 1, belief1_upper);
	}else{
		double prior1_upper = this->GetBliefBound(1, 1) - (belief1_upper - this->GetPriorBound(1, 1));
		if(prior1_upper < this->GetPriorBound(1, 1)) this->SetPriorBound(1, 1, prior1_upper);
	}
}

void FactorNode::BoundPropagation() {
	for(int i = 0; i < 2; i++){
		double msg0_upper = 0, msg0_lower = 0, msg1_upper = 0, msg1_lower = 0;
		double dij = this->func->GetValue(0, 0) - this->func->GetValue(0, 1);
		VariableNode * vi = (VariableNode*)neighbor[i];
		VariableNode * vj = (VariableNode*)neighbor[1 - i];
		if(dij > 0){ // homophily
			msg1_upper += 0.5 * dij * ((vj->GetBliefBound(1, 1) - vj->GetMsgBound(id, 1, 0)) - (vj->GetBliefBound(0, 0) - vj->GetMsgBound(id, 1, 1)));
			msg1_lower += 0.5 * dij * ((vj->GetBliefBound(1, 0) - vj->GetMsgBound(id, 1, 1)) - (vj->GetBliefBound(0, 1) - vj->GetMsgBound(id, 0, 0)));
			msg0_upper += 0.5 * dij * ((vj->GetBliefBound(0, 1) - vj->GetMsgBound(id, 0, 0)) - (vj->GetBliefBound(1, 0) - vj->GetMsgBound(id, 1, 1)));
			msg0_lower += 0.5 * dij * ((vj->GetBliefBound(0, 0) - vj->GetMsgBound(id, 0, 1)) - (vj->GetBliefBound(1, 1) - vj->GetMsgBound(id, 1, 0)));
		}else{ // anti-homophily
			msg1_upper += 0.5 * dij * ((vj->GetBliefBound(1, 0) - vj->GetMsgBound(id, 1, 1)) - (vj->GetBliefBound(0, 1) - vj->GetMsgBound(id, 0, 0)));
			msg1_lower += 0.5 * dij * ((vj->GetBliefBound(1, 1) - vj->GetMsgBound(id, 1, 0)) - (vj->GetBliefBound(0, 0) - vj->GetMsgBound(id, 0, 1)));
			msg0_upper += 0.5 * dij * ((vj->GetBliefBound(0, 0) - vj->GetMsgBound(id, 0, 1)) - (vj->GetBliefBound(1, 1) - vj->GetMsgBound(id, 1, 0)));
			msg0_lower += 0.5 * dij * ((vj->GetBliefBound(0, 1) - vj->GetMsgBound(id, 0, 0)) - (vj->GetBliefBound(1, 0) - vj->GetMsgBound(id, 1, 1)));
		}
		if(msg0_lower > vi->GetMsgBound(id, 0, 0)) vi->SetMsgBound(id, 0, 0, msg0_lower);
		if(msg0_upper < vi->GetMsgBound(id, 0, 1)) vi->SetMsgBound(id, 0, 1, msg0_upper);
		if(msg1_lower > vi->GetMsgBound(id, 1, 0)) vi->SetMsgBound(id, 1, 0, msg1_lower);
		if(msg1_upper < vi->GetMsgBound(id, 1, 1)) vi->SetMsgBound(id, 1, 1, msg1_upper);
	}
}

/************************************************
 * Bound propagation setters and getters
 ************************************************/
double VariableNode::GetPriorBound(int label, int is_upper){
	return this->state_factor[label * num_label + is_upper];
}

double VariableNode::GetBliefBound(int label, int is_upper){
	return this->belief_bound[label * num_label + is_upper];
}

void VariableNode::SetPriorBound(int label, int is_upper, double value){
	this->prior_bound[label * num_label + is_upper] = value;
}

void VariableNode::SetBliefBound(int label, int is_upper, double value){
	this->belief_bound[label * num_label + is_upper] = value;
}

/**************************************************
 FactorGraph
**************************************************/

void FactorGraph::InitGraph(int n, int m, int num_label)
{
	this->labeled_given = false;
	this->n = n;
	this->m = m;
	this->num_label = num_label;
	this->num_node = n + m;
	
	var_node = new VariableNode[n];
	factor_node = new FactorNode[m];

	int p_node_id = 0;

	p_node = new Node*[n + m];
	for (int i = 0; i < n; i++)
	{
		var_node[i].id = p_node_id;
		p_node[p_node_id ++] = &var_node[i];
		
		var_node[i].Init( num_label );
	}

	for (int i = 0; i < m; i++)
	{
		factor_node[i].id = p_node_id;
		p_node[p_node_id ++] = &factor_node[i];

		factor_node[i].Init( num_label );
	}

	factor_node_used = 0;
}

// TODO server
void FactorGraph::AddEdge(int a, int b, FactorFunction* func, int edge_type)
{
	// AddEdge can be called at most m times
	if (factor_node_used == m) return;

	factor_node[factor_node_used].func = func;
    // TODO server
    factor_node[factor_node_used].edge_type = edge_type;
	
	factor_node[factor_node_used].AddNeighbor( &var_node[a] );
	factor_node[factor_node_used].AddNeighbor( &var_node[b] );

	var_node[a].AddNeighbor( &factor_node[factor_node_used] );
	var_node[b].AddNeighbor( &factor_node[factor_node_used] );

	factor_node_used++;
}

void FactorGraph::ClearDataForSumProduct()
{   
	for (int i = 0; i < n; i++)
	{
		MatrixUtil::DoubleArrFill(var_node[i].state_factor, num_label, 1.0 / num_label);          
	}

	for (int i = 0; i < num_node; i++)
	{
		for (int t = 0; t < p_node[i]->neighbor.size(); t++)
		{
			MatrixUtil::DoubleArrFill(p_node[i]->belief[t], num_label, 1.0 / num_label);
		}
	}
}

void FactorGraph::ClearDataForMaxSum()
{
	for (int i = 0; i < n; i++)
	{
		MatrixUtil::DoubleArrFill(var_node[i].state_factor, num_label, 1.0 / num_label);
	}
	for (int i = 0; i < num_node; i++)
	{
		for (int t = 0; t < p_node[i]->neighbor.size(); t++)
		{
			for (int y = 0; y < num_label; y++)
				p_node[i]->belief[t][y] = 1.0 / num_label;
		}
	}
}

void FactorGraph::GenPropagateOrder()
{
	bool* mark = new bool[num_node];
	bfs_node = new Node*[num_node];

	for (int i = 0; i < num_node; i++)
		mark[i] = false;

	int head = 0, tail = -1;
	for (int i = 0; i < num_node; i++)
	{
		if (! mark[i])
		{
			entry.push_back( p_node[i] );
			bfs_node[++tail] = p_node[i];
			mark[p_node[i]->id] = 1;

			while (head <= tail)
			{
				Node* u = bfs_node[head++];
				for (vector<Node*>::iterator it = u->neighbor.begin(); it != u->neighbor.end(); it++)
					if (! mark[(*it)->id] )
					{
						bfs_node[++tail] = *it;
						mark[(*it)->id] = 1;
					}
			}
		}
	}

	delete[] mark;
}

void FactorGraph::BeliefPropagation(int max_iter)
{    
	int start, end, dir;

	converged = false;
	for (int iter = 0; iter < max_iter; iter++)
	{
		diff_max = 0.0;

		if (iter % 2 == 0)
			start = num_node - 1, end = -1, dir = -1;
		else
			start = 0, end = num_node, dir = +1;

		for (int p = start; p != end; p += dir)
		{
			bfs_node[p]->BeliefPropagation(&diff_max, this->labeled_given);
		}

		if (diff_max < 1e-6) break;
	}
}

void FactorGraph::CalculateMarginal()
{
	for (int i = 0; i < n; i++)
	{
		double sum_py = 0.0;
		for (int y = 0; y < num_label; y++)
		{
			var_node[i].marginal[y] = var_node[i].state_factor[y];
			for (int t = 0; t < var_node[i].neighbor.size(); t++)
				var_node[i].marginal[y] *= var_node[i].belief[t][y];
			sum_py += var_node[i].marginal[y];
		}
		for (int y = 0; y < num_label; y++)
		{
			var_node[i].marginal[y] /= sum_py;
		}
	}

	for (int i = 0; i < m; i++)
	{
		double sump = 0.0;
		for (int a = 0; a < num_label; a++)
			for (int b = 0; b < num_label; b++)
			{
				factor_node[i].marginal[a][b] +=
					factor_node[i].belief[0][a]
					* factor_node[i].belief[1][b]
					* factor_node[i].func->GetValue(a, b);
				sump += factor_node[i].marginal[a][b];
				if(sump == 0.0)
					sump += 1e-5;
			}
		for (int a = 0; a < num_label; a++)
			for (int b = 0; b < num_label; b++)
				factor_node[i].marginal[a][b] /= sump;
	}
}

void FactorGraph::MaxSumPropagation(int max_iter)
{
	int start, end, dir;

	converged = false;
	for (int iter = 0; iter < max_iter; iter++)
	{
		diff_max = 0;

		if (iter % 2 == 0)
			start = num_node - 1, end = -1, dir = -1;
		else
			start = 0, end = num_node, dir = +1;

		for (int p = start; p != end; p += dir)
		{
			bfs_node[p]->MaxSumPropagation(&diff_max, labeled_given);
		}

		if (diff_max < 1e-6) break;
	}
}

void FactorGraph::ClearDataForBoundPropagation(){
	for(int i = 0; i < n; i++){
		MatrixUtil::DoubleArrFillBound(var_node[i].belief_bound);
		MatrixUtil::DoubleArrFillBound(var_node[i].prior_bound);
		if(var_node[i].label_type == Enum::KNOWN_LABEL){
			int label = var_node[i].y;
			var_node[i].SetBliefBound(label, 1, 1.0);
			var_node[i].SetBliefBound(label, 0, 1.0);
			var_node[i].SetBliefBound(1 - label, 0, 0.0);
			var_node[i].SetBliefBound(1 - label, 1, 0.0);
		}
	}
	for(int i = 0; i < num_node; i ++){
		for(int t = 0; t < p_node[i]->neighbor.size(); t ++){
			MatrixUtil::DoubleArrFillBound(p_node[i]->msg_bound[t]);
		}
	}
}

void FactorGraph::BoundPropagation(int max_iter){
	int start, end, dir;
	converged = false;
	for(int iter = 0; iter < max_iter; iter++){
		if(iter % 2 == 0)
			start = num_node - 1, end = -1, dir = -1;
		else
			start = 0, end = num_node, dir = 1;
		for (int p = start; p != end; p += dir){
			bfs_node[p]->BoundPropagation();
		}
		// int num_of_anchor_nodes = this->FindAnchorNode();
		// printf("[iter %d] anchor nodes: %d\n", iter, num_of_anchor_nodes);
	}
}

// int FactorGraph::FindAnchorNode(){
//	int num_var_node = this->n;
//	for(int i = 0; i < num_var_node; i++){
//		double b0_lower = this->var_node[i].GetBliefBound(0, 0);
//		double b1_lower = this->var_node[i].GetBliefBound(1, 0);
//		double b0_upper = this->var_node[i].GetBliefBound(0, 1);
//		double b1_upper = this->var_node[1].GetBliefBound(1, 1);
//		if(b0_lower > 0.5 || b1_lower > 0.5){
//			this->anchor_nodes.insert(i);
//		} else if(b0_upper < 0.5 || b1_upper < 0.5){
//			this->anchor_nodes.insert(i);
//		}
//	}
//	return (int)this->anchor_nodes.size();
//}

void FactorGraph::Clean()
{
	if (var_node) delete[] var_node;
	if (factor_node) delete[] factor_node;
	if (p_node) delete[] p_node;
	if (bfs_node) delete[] bfs_node;
}
