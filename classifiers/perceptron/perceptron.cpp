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
	accuracy			 = 0;
	errorRate			 = 0;
	precision			 = 0;
	sensitivity			 = 0;
	specificity			 = 0;
}

void Perceptron::config(int numNeurons, int numSynapsesPerNeuron, int numClasses)
{
	this->numNeurons = numNeurons;
	this->numSynapsesPerNeuron = numSynapsesPerNeuron;
	neurons.resize(numNeurons);
	output.resize(numNeurons);
	outputError.resize(numNeurons);

	for(int i = 0; i < numNeurons; i++)
		neurons[i].init(numSynapsesPerNeuron);

	confusionMatrix.resize(numClasses);
	for(int j = 0; j < numClasses; j++)
		confusionMatrix[j].resize(numClasses);
}

void Perceptron::trainingConfig(int numEpochs, double learningRate)
{
	this->numEpochs = numEpochs;
	this->learningRate = learningRate;
}

void Perceptron::training(std::vector< std::vector<double> >& data, std::vector< std::vector<int> >& target)
{
	int numExamples = data.size();
	int realClass, predClass;
	int cnt = 0;

	for(int epoch = 0; epoch < numEpochs; epoch++)
	{
		for(int example = 0; example < numExamples; example++)
		{
			perceptronOutput(data[example]);
			getErrorsAndAdjust(epoch, target[example], data[example]);

			// Preencher Matrix de confusao na ultima epoca de treinamento.
			if( epoch == (numEpochs - 1) )
			{
				realClass = 0;
				predClass = 0;
				for(int i = 0; i < numNeurons; i++)
				{
					if( target[example][i] == 1 )
						realClass = i;

					if( output[i] == 1 )
						predClass = i;
				}
				confusionMatrix[realClass][predClass]++;
			}
		}
		cnt++;
	}
	cout << endl << "Quantidade de epocas: " << cnt << endl;
	cout << "Numero de atualizacoes de pesos: " << changeWeights << endl;
}

void Perceptron::perceptronOutput(std::vector<double>& input)
{
	int pot = 0;

	for(int n = 0; n < numNeurons; n++)
	{
		pot = neurons[n].activationPotencial(input);

		if(pot >= 0)
			output[n] = 1;
		else
			output[n] = 0;
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

void Perceptron::computePerformanceMetrics()
{
	float TP, FN, FP, TN;
	float P, N;

	TP = confusionMatrix[0][0];
	FN = confusionMatrix[1][0];
	FP = confusionMatrix[0][1];
	TN = confusionMatrix[1][1];

	P = TP + FN;
	N = FP + TN;

	accuracy = (TP + TN) / (P + N);
	errorRate = 1 - accuracy;
	precision = TP / (TP + FP);
	sensitivity = TP / P;
	specificity = TN / N;

	cout << endl << "------ Metricas de Desempenho ------" << endl;
	cout << "Accuracy = " << accuracy << endl;
	cout << "Error Rate = " << errorRate << endl;
	cout << "Precision = " << precision << endl;
	cout << "Sensitivity = " << sensitivity << endl;
	cout << "Specificity = " << specificity << endl;
}

void Perceptron::printWeights()
{
	for(unsigned int n = 0; n < neurons.size(); n++)
	{
		cout << "Neuron " << n << endl;
		for(unsigned int w = 0; w < neurons[n].weights.size(); w++)
			cout << "	W" << w << " = " << neurons[n].weights[w] << endl;
		cout << endl;
	}
}
