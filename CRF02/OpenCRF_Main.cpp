#include "Config.h"
#include "DataSet.h"
#include "CRFModel.h"

// OpenCRF.exe -est -niter 100 -gradientstep 0.001 -trainfile example.txt -dstmodel model.txt
int main(int argc, char* argv[])
{
//    char str[] = "OpenCRF.exe -est -niter 200 -gradientstep 0.01 -trainfile /Users/athena/CLionProjects/ForCRFNew/CRF/OpenCRF/testdata/example.txt -dstmodel model.txt -node_alpha 0.001 -edge_alpha 0.002";
//    argc = split(str, argv);

	// Load Configuartion
	Config* conf = new Config();
	if (! conf->LoadConfig(argc, argv))
	{
		conf->ShowUsage();
		exit( 0 );
	}

	CRFModel *model = new CRFModel();

	if (conf->task == "-est")
	{
		model->Estimate(conf);
	}
	else if (conf->task == "-estc")
	{
		model->EstimateContinue(conf);
	}
	else if (conf->task == "-inf")
	{
		model->Inference(conf);
	}
	else
	{
		Config::ShowUsage();
	}

	return 0;
}

