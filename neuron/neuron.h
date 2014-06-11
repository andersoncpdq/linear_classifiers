/*
 * neuron.h
 *
 *  Created on: 08/06/2014
 *      Author: anderson
 */

#ifndef NEURON_H_
#define NEURON_H_

#include <vector>
#include <cstdlib>
#include </home/anderson/workspace/linearClassifiers/mersenne/mtwist.h>

class Neuron
{
public:
	// Atributos
	int numSynapses;
	double bias;
	double biasOld;
	std::vector<double> weights;
	std::vector<double> weightsOld;

	// Metodos
	Neuron();
	void init(int numSynapses);
	void weightsInit();
	double activationPotencial(std::vector<double>& input);
};

#endif /* NEURON_H_ */
