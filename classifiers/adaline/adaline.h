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

	// Métodos
	Adaline();
	void config(int numNeurons, int numSynapsesPerNeuron, int numClasses);
	void trainingConfig(int numEpochs, double learningRate, double acceptableError);
	void training(std::vector< std::vector<double> >& data, std::vector< std::vector<int> >& target);
	void computePotentialAndError(std::vector<double>& input, std::vector<int>& target);
	void getError(std::vector<int>& target);
	void adjustWeights(std::vector<double>& input, std::vector<int>& target);
	void getLeastMeanSquare(int epoch, std::vector< std::vector<double> >& data, std::vector< std::vector<int> >& target);
};

#endif /* ADALINE_H_ */
