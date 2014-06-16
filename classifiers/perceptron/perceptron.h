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

	// Metodos
	Perceptron();
	void config(int numNeurons, int numSynapsesPerNeuron, int numFolds);
	void trainingConfig(int numEpochs, double learningRate);
	void training(std::vector< std::vector<double> >& data, std::vector< std::vector<int> >& target);
	void perceptronOutput(std::vector<double>& input);
	void getErrorsAndAdjust(int epoch, std::vector<int>& target, std::vector<double>& input);
	void adjustWeights(int indexNeuron, std::vector<double>& input);
	void operation(std::vector< std::vector<double> >& data, std::vector< std::vector<int> >& target);
	void computePerformanceMetrics(int cntFolds, int numFolds);
};

#endif /* PERCEPTRON_H_ */
