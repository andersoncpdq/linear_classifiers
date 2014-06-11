/*
 * neuron.cpp
 *
 *  Created on: 08/06/2014
 *      Author: anderson
 */

#include "neuron.h"

Neuron::Neuron()
{
	bias 		= 0;
	biasOld 	= 0;
	numSynapses = 0;
}

void Neuron::init(int numSynapses)
{
	this->numSynapses = numSynapses;
	weights.resize(numSynapses);
	weightsOld.resize(numSynapses);

	// Inicializa-se os pesos
	weightsInit();
}

void Neuron::weightsInit()
{
	// seed
	mt_seed();

	for(int i = 0; i < numSynapses; i++)
	{
		weights[i] = mt_ldrand(); // 0.0 - 1.0
		weightsOld[i] = weights[i];
	}

	bias = mt_ldrand();
	biasOld = bias;
}

double Neuron::activationPotencial(std::vector<double>& input)
{
	double potential = bias;

	for(int i = 0; i < numSynapses; i++)
		potential += input[i] * weights[i];

	return potential;
}
