#include "Config.h"
#include "DataSet.h"
#include "CRFModel.h"

int split(char sentence[], char* argv[]){
    const char* sep = " ";
    char* a = strtok(sentence, sep);
    int index = 0;
    while(a){
//        printf("%s\n", a);
        argv[index++] = a;
        a = strtok(NULL, sep);
    }
    return index;
}

// OpenCRF.exe -est -niter 100 -gradientstep 0.001 -trainfile example.txt -dstmodel model.txt
int main(int argc, char* argv[])
{
    char str[] = "OpenCRF.exe -est -niter 200 -gradientstep 0.01 -trainfile /Users/athena/CLionProjects/ForCRFCompare/CRF/OpenCRF/testdata/cora_123class.txt -dstmodel model.txt";
    argc = split(str, argv);

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

