/*
 * main.cpp
 *
 *  Created on: 08/06/2014
 *      Author: anderson
 */

#include <fstream>
#include <iomanip>

#include "classifiers/perceptron/perceptron.h"
#include "classifiers/adaline/adaline.h"

std::vector<double> mean(std::vector< std::vector<double> >& data);
std::vector<double> stdDeviation(std::vector< std::vector<double> >& data, std::vector<double>& meanMatrix);
void zscore(std::vector< std::vector<double> >& data, std::vector<double>& meanMatrix, std::vector<double>& stdMatrix);

void fileInput(const char dataset[], std::vector< std::vector<double> >& data, std::vector< std::vector<int> >& target);
void fileOutputWeights(const char dataset[], std::vector<Neuron>);
void fileOutputConfusionMatrix(const char dataset[], std::vector< std::vector<int> >);

int main()
{
    /*
     *	1 - Perceptron Simples
     *	2 - Adaline
     */
    int idClassifier = 1;

	int numEpochs = 100;
	double learningRate = 0.1;
	double acceptableError = 0.01;

	std::vector< std::vector<double> > trainingData;
	std::vector< std::vector<int> > trainingTarget;
	std::vector<double> meanMatrix;
	std::vector<double> stdMatrix;

	//fileInput("datasets/iris.data", trainingData, trainingTarget);
	//fileInput("datasets/wine.data", trainingData, trainingTarget);
	//fileInput("datasets/dermatology.data", trainingData, trainingTarget);
	fileInput("datasets/column.data", trainingData, trainingTarget);

	// normalizacao z-score
    meanMatrix = mean(trainingData);
    stdMatrix = stdDeviation(trainingData, meanMatrix);
    zscore(trainingData, meanMatrix, stdMatrix);

    int numInputs = trainingData[0].size();
    int numOutputs = trainingTarget[0].size();

    cout << "Classificadores Lineares" << endl << endl;

    if(idClassifier == 1)
    {
		cout << "Perceptron Simples" << endl << endl;
		cout << " A G U A R D E   O   P R O C E S S A M E N T O" << endl;

		Perceptron perceptron;
		perceptron.config(2, numInputs, numOutputs);
		perceptron.trainingConfig(numEpochs, learningRate);
		perceptron.training(trainingData, trainingTarget);
		perceptron.computePerformanceMetrics();

		fileOutputWeights("weights.txt", perceptron.neurons);
		fileOutputConfusionMatrix("confusionMatrix.txt", perceptron.confusionMatrix);
    }
    else if(idClassifier == 2)
    {
		cout << "Adaline" << endl << endl;
		cout << " A G U A R D E   O   P R O C E S S A M E N T O" << endl;

		Adaline adaline;
		adaline.config(2, numInputs, numOutputs);
		adaline.trainingConfig(numEpochs, learningRate, acceptableError);
		adaline.training(trainingData, trainingTarget);
		adaline.operation(trainingData, trainingTarget);

		fileOutputWeights("weights.txt", adaline.neurons);
		fileOutputConfusionMatrix("confusionMatrix.txt", adaline.confusionMatrix);
    }

    cout << endl << "Processamento Concluido!" << endl;
	return 0;
}

std::vector<double> mean(std::vector< std::vector<double> >& data)
{
	double sum = 0;
	int numRows = data.size();
	int numCols = data[0].size();

	std::vector<double> mean;
	mean.resize(numCols);
	mean.clear();

	for(int i = 0; i < numCols; i++)
	{
		for(int j = 0; j < numRows; j++)
			sum += data[j][i];

		mean.push_back(sum/numRows);
		sum = 0;
	}
	return mean;
}

std::vector<double> stdDeviation(std::vector< std::vector<double> >& data, std::vector<double>& meanMatrix)
{
	double sum = 0;
	int numRows = data.size();
	int numCols = data[0].size();

	std::vector<double> stdDev;
	stdDev.resize(numCols);
	stdDev.clear();

	for(int i = 0; i < numCols; i++)
	{
		for(int j = 0; j < numRows; j++)
			sum += pow( (data[j][i] - meanMatrix[i]), 2);

		stdDev.push_back( sqrt( sum / (numRows - 1) ) );
		sum = 0;
	}

	return stdDev;
}

void zscore(std::vector< std::vector<double> >& data, std::vector<double>& meanMatrix, std::vector<double>& stdMatrix)
{
	int numRows = data.size();
	int numCols = data[0].size();

	for(int i = 0; i < numCols; i++)
	{
		for(int j = 0; j < numRows; j++)
			data[j][i] = ( ( data[j][i] - meanMatrix[i] ) / stdMatrix[i] );
	}
}

void fileInput(const char dataset[], std::vector< std::vector<double> >& data, std::vector< std::vector<int> >& target)
{
	int numExamples;
	int numInputs;
	int numTargets;
	const int TITLE_LENGHT = 100;
	char title[TITLE_LENGHT];

    // abertura do arquivo atraves do construtor ifstream.
    ifstream inFile(dataset, ios::in);

    // termina o programa caso o arquivo nao possa ser aberto.
    if( !inFile )
    {
        cerr << "O arquivo de treino nao pode ser aberto!" << endl;
        exit(1);
    }

    inFile.getline(title, TITLE_LENGHT, '\n');

    inFile >> numExamples;
    inFile >> numInputs;
    inFile >> numTargets;

    inFile.getline(title, TITLE_LENGHT, '\n');
    inFile.getline(title, TITLE_LENGHT, '\n');

    data.resize(numExamples);
    target.resize(numExamples);

    for(int i = 0; i < numExamples; i++)
    {
        data[i].resize(numInputs);
        target[i].resize(numTargets);
    }

    for(unsigned int i = 0; i < data.size(); i++)
    {
    	//cout << i << ": ";
        for (unsigned int j = 0; j < data[i].size(); j++)
        {
            inFile >> data[i][j];
            //cout << data[i][j] << " ";
        }

        //cout << "output: ";
        for (unsigned int j = 0; j < target[i].size(); j++)
        {
            inFile >> target[i][j];
            //cout << target[i][j] << " ";
        }
        //cout << endl;
    }
    inFile.close();
}

void fileOutputWeights(const char dataset[], std::vector<Neuron> neurons)
{
    ofstream outFile(dataset, ios::out);

    if (!outFile)
    {
    	cerr << "O log dos pesos não pode ser criado!" << endl;
    	exit(1);
    }

	for(unsigned int n = 0; n < neurons.size(); n++)
	{
		for(unsigned int w = 0; w < neurons[n].weights.size(); w++)
			outFile << neurons[n].weights[w] << "\t";
		outFile << "\n";
	}
}

void fileOutputConfusionMatrix(const char dataset[], std::vector< std::vector<int> > confusionMatrix)
{
    ofstream outFile(dataset, ios::out);

    if (!outFile)
    {
    	cerr << "O log da matriz de confusao não pode ser criado!" << endl;
    	exit(1);
    }

    outFile << "##### Matriz de Confusao #####" << "\n\n";
    outFile << "Real Predita" << "\n";
	for(unsigned int i = 0; i < confusionMatrix.size(); i++)
	{
		for(unsigned int j = 0; j < confusionMatrix[i].size(); j++)
			outFile << " [" << i << "]   [" << j << "]  =  " << confusionMatrix[i][j] << "\n";
		outFile << "\n";
	}
}
