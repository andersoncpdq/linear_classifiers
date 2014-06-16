/*
 * perceptron.cpp
 *
 *  Created on: 08/06/2014
 *      Author: anderson
 */

#include "perceptron.h"

Perceptron::Perceptron()
{
	numEpochs 			 = 0;
	numNeurons 			 = 0;
	numSynapsesPerNeuron = 0;
	learningRate 		 = 0;
	changeWeights		 = 0;
	meanAccuracy		 = 0;
	meanErrorRate		 = 0;
	meanPrecision		 = 0;
	meanSensivity		 = 0;
	meanSpecificity		 = 0;
}

void Perceptron::config(int numNeurons, int numSynapsesPerNeuron, int numFolds)
{
	this->numNeurons = numNeurons;
	this->numSynapsesPerNeuron = numSynapsesPerNeuron;
	neurons.resize(numNeurons);
	output.resize(numNeurons);
	outputError.resize(numNeurons);

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

void Perceptron::trainingConfig(int numEpochs, double learningRate)
{
	this->numEpochs = numEpochs;
	this->learningRate = learningRate;
}

void Perceptron::training(std::vector< std::vector<double> >& data, std::vector< std::vector<int> >& target)
{
	int numExamples = data.size();
	int cntEpochs = 0;

	for(int epoch = 0; epoch < numEpochs; epoch++)
	{
		for(int example = 0; example < numExamples; example++)
		{
			perceptronOutput(data[example]);
			getErrorsAndAdjust(epoch, target[example], data[example]);
		}
		cntEpochs++;
	}
	cout << endl << "Quantidade de epocas: " << cntEpochs << endl;
	cout << "Numero de atualizacoes de pesos: " << changeWeights << endl;
}

void Perceptron::perceptronOutput(std::vector<double>& input)
{
	double pot = 0;

	for(int n = 0; n < numNeurons; n++)
	{
		pot = neurons[n].activationPotencial(input);

		if(pot >= 0.0)
			output[n] = 1;
		else
			output[n] = -1;
	}
}

void Perceptron::getErrorsAndAdjust(int epoch, std::vector<int>& target, std::vector<double>& input)
{
	for(int n = 0; n < numNeurons; n++)
	{
		outputError[n] = target[n] - output[n];

		if(outputError[n] != 0)
		{
			changeWeights++;
			adjustWeights(n, input);
		}
	}
}

void Perceptron::adjustWeights(int indexNeuron, std::vector<double>& input)
{
	double temp = 0;

	for(int w = 0; w < numSynapsesPerNeuron; w++)
	{
		temp = neurons[indexNeuron].weights[w];
		neurons[indexNeuron].weights[w] += learningRate * outputError[indexNeuron] * input[w];
		neurons[indexNeuron].weightsOld[w] = temp;
	}

	temp = neurons[indexNeuron].bias;
	neurons[indexNeuron].bias += learningRate * outputError[indexNeuron] * 1;
	neurons[indexNeuron].biasOld = temp;
}

void Perceptron::operation(std::vector< std::vector<double> >& data, std::vector< std::vector<int> >& target)
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

void Perceptron::computePerformanceMetrics(int cntFolds, int numFolds)
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
