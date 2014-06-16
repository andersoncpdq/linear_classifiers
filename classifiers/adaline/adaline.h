/*
 * adaline.h
 *
 *  Created on: 10/06/2014
 *      Author: anderson
 */

#ifndef ADALINE_H_
#define ADALINE_H_

#include <iostream>
#include <math.h>

#include "/home/anderson/workspace/linearClassifiers/neuron/neuron.h"

using namespace std;

class Adaline
{
public:
	// Atributos
	int numEpochs;
	int numNeurons;
	int numSynapsesPerNeuron;
	double learningRate;
	double acceptableError;
	std::vector<Neuron> neurons;
	std::vector<double> potential;
	std::vector<double> error;
	std::vector<int> output;
	std::vector<double> leastMeanSquare;

	std::vector< std::vector<int> > confusionMatrix;
	std::vector< std::vector<int> > confusionMatrixTotal;

	std::vector<double> accuracy;
	std::vector<double> errorRate;
	std::vector<double> precision;
	std::vector<double> sensitivity;
	std::vector<double> specificity;

	double meanAccuracy;
	double meanErrorRate;
	double meanPrecision;
	double meanSensivity;
	double meanSpecificity;

	// Métodos
	Adaline();
	void config(int numNeurons, int numSynapsesPerNeuron, int numFolds);
	void trainingConfig(int numEpochs, double learningRate, double acceptableError);
	void training(std::vector< std::vector<double> >& data, std::vector< std::vector<int> >& target);
	void computePotentialAndError(int epoch, std::vector<double>& input, std::vector<int>& target);
	void adjustWeights(std::vector<double>& input, std::vector<int>& target);
	void operation(std::vector< std::vector<double> >& data, std::vector< std::vector<int> >& target);
	void computePerformanceMetrics(int cntFolds, int numFolds);
};

#endif /* ADALINE_H_ */
