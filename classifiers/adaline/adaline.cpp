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
}

void Adaline::config(int numNeurons, int numSynapsesPerNeuron, int numClasses)
{
	this->numNeurons = numNeurons;
	this->numSynapsesPerNeuron = numSynapsesPerNeuron;
	neurons.resize(numNeurons);
	potential.resize(numNeurons);
	output.resize(numNeurons);
	error.resize(numNeurons);

	for(int i = 0; i < numNeurons; i++)
		neurons[i].init(numSynapsesPerNeuron);

	confusionMatrix.resize(numClasses);
	for(int j = 0; j < numClasses; j++)
		confusionMatrix[j].resize(numClasses);
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
				output[n] = 0;

			if( target[example][n] == 1 )
				realClass = n;

			if( output[n] == 1 )
				predClass = n;
		}
		confusionMatrix[realClass][predClass]++;
	}
}
