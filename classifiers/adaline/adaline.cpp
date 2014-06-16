/*
 * adaline.cpp
 *
 *  Created on: 10/06/2014
 *      Author: anderson
 */

#include "adaline.h"

Adaline::Adaline()
{
	numEpochs 			 = 0;
	numNeurons 			 = 0;
	numSynapsesPerNeuron = 0;
	learningRate 		 = 0;
	acceptableError		 = 0;
	meanAccuracy		 = 0;
	meanErrorRate		 = 0;
	meanPrecision		 = 0;
	meanSensivity		 = 0;
	meanSpecificity		 = 0;
}

void Adaline::config(int numNeurons, int numSynapsesPerNeuron, int numFolds)
{
	this->numNeurons = numNeurons;
	this->numSynapsesPerNeuron = numSynapsesPerNeuron;
	neurons.resize(numNeurons);
	potential.resize(numNeurons);
	output.resize(numNeurons);
	error.resize(numNeurons);

	for(int i = 0; i < numNeurons; i++)
		neurons[i].init(numSynapsesPerNeuron);

	confusionMatrix.resize(2);
	confusionMatrixTotal.resize(2);
	for(int j = 0; j < 2; j++)
	{
		confusionMatrix[j].resize(2);
		confusionMatrixTotal[j].resize(2);
	}

	accuracy.resize(numFolds);
	errorRate.resize(numFolds);
	precision.resize(numFolds);
	sensitivity.resize(numFolds);
	specificity.resize(numFolds);
}

void Adaline::trainingConfig(int numEpochs, double learningRate, double acceptableError)
{
	this->numEpochs = numEpochs;
	this->learningRate = learningRate;
	this->acceptableError = acceptableError;
	leastMeanSquare.resize(numEpochs);
}

void Adaline::training(std::vector< std::vector<double> >& data, std::vector< std::vector<int> >& target)
{
	int numExamples = data.size();
	int cntEpochs = 0;

	for(int epoch = 0; epoch < numEpochs; epoch++)
	{
		for(int example = 0; example < numExamples; example++)
		{
			computePotentialAndError(epoch, data[example], target[example]);
			adjustWeights(data[example], target[example]);
		}
		cntEpochs++;

		leastMeanSquare[epoch] /= numExamples;

		//cout << "LMS[" << epoch << "] = " << leastMeanSquare[epoch] << endl;

		if( leastMeanSquare[epoch] <= acceptableError )
			break;
	}
	cout << endl << "Quantidade de epocas: " << cntEpochs << endl;
}

void Adaline::computePotentialAndError(int epoch, std::vector<double>& input, std::vector<int>& target)
{
	double ms = 0;

	for(int n = 0; n < numNeurons; n++)
	{
		potential[n] = neurons[n].activationPotencial(input);
		error[n] = target[n] - potential[n];

		ms += pow(error[n], 2);
	}

	leastMeanSquare[epoch] += ms;
}

void Adaline::adjustWeights(std::vector<double>& input, std::vector<int>& target)
{
	double temp = 0;

	for(int n = 0; n < numNeurons; n++)
	{
		for(int w = 0; w < numSynapsesPerNeuron; w++)
		{
			temp = neurons[n].weights[w];
			neurons[n].weights[w] += learningRate * error[n] * input[w];
			neurons[n].weightsOld[w] = temp;
		}

		temp = neurons[n].bias;
		neurons[n].bias += learningRate * error[n] * 1;
		neurons[n].biasOld = temp;
	}
}

void Adaline::operation(std::vector< std::vector<double> >& data, std::vector< std::vector<int> >& target)
{
	int numExamples = data.size();
	int realClass, predClass;
	double pot = 0;

	for(int example = 0; example < numExamples; example++)
	{
		realClass = 0;
		predClass = 0;
		for(int n = 0; n < numNeurons; n++)
		{
			pot = neurons[n].activationPotencial(data[example]);

			if(pot >= 0.0)
				output[n] = 1;
			else
				output[n] = -1;

			if(target[example][n] == 1)
				realClass = 0;
			else
				realClass = 1;

			if(output[n] == 1)
				predClass = 0;
			else
				predClass = 1;

			confusionMatrix[realClass][predClass]++;
			confusionMatrixTotal[realClass][predClass]++;
		}
	}
}

void Adaline::computePerformanceMetrics(int cntFolds, int numFolds)
{
	double TP, FN, FP, TN;
	double P, N;

	TP = confusionMatrix[0][0];
	FN = confusionMatrix[1][0];
	FP = confusionMatrix[0][1];
	TN = confusionMatrix[1][1];

	P = TP + FN;
	N = FP + TN;

	if( (P + N) != 0 )
		accuracy[cntFolds] = ( (TP + TN) / (P + N) );

	errorRate[cntFolds] = (1 - accuracy[cntFolds]);

	if( (TP + FP) != 0 )
		precision[cntFolds] = ( TP / (TP + FP) );

	if(P != 0)
		sensitivity[cntFolds] = ( TP / P );

	if(N != 0)
		specificity[cntFolds] = ( TN / N );

	if(cntFolds == 4)
	{
		double acc = 0, err = 0, pre = 0, sen = 0, spe = 0;
		for(int i = 0; i < numFolds; i++)
		{
			acc += accuracy[i];
			err += errorRate[i];
			pre += precision[i];
			sen += sensitivity[i];
			spe += specificity[i];
		}
		meanAccuracy = acc / numFolds;
		meanErrorRate = err / numFolds;
		meanPrecision = pre / numFolds;
		meanSensivity = sen / numFolds;
		meanSpecificity = spe / numFolds;
	}
}
