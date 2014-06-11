/*
 * perceptron.h
 *
 *  Created on: 08/06/2014
 *      Author: anderson
 */

#ifndef PERCEPTRON_H_
#define PERCEPTRON_H_

#include <iostream>
#include <math.h>

#include "/home/anderson/workspace/linearClassifiers/neuron/neuron.h"

using namespace std;

class Perceptron
{
public:
	// Atributos
	int numEpochs;
	int numNeurons;
	int numSynapsesPerNeuron;
	double learningRate;
	int changeWeights;
	std::vector<Neuron> neurons;
	std::vector<int> output;
	std::vector<int> outputError;

	std::vector< std::vector<int> > confusionMatrix;
	float accuracy;
	float errorRate;
	float precision;
	float sensitivity;
	float specificity;

	// Metodos
	Perceptron();
	void config(int numNeurons, int numSynapsesPerNeuron, int numClasses);
	void trainingConfig(int numEpochs, double learningRate);
	void training(std::vector< std::vector<double> >& data, std::vector< std::vector<int> >& target);
	void perceptronOutput(std::vector<double>& input);
	void getErrorsAndAdjust(int epoch, std::vector<int>& target, std::vector<double>& input);
	void adjustWeights(int indexNeuron, std::vector<double>& input);
	void computePerformanceMetrics();
	void printWeights();
};

#endif /* PERCEPTRON_H_ */
