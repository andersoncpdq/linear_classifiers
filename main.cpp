/*
 * main.cpp
 *
 *  Created on: 08/06/2014
 *      Author: anderson
 */

#include <fstream>
#include <iomanip>
#include <string.h>
#include <sstream>

#include "classifiers/perceptron/perceptron.h"
#include "classifiers/adaline/adaline.h"

std::vector<double> mean(std::vector< std::vector<double> >& data);
std::vector<double> stdDeviation(std::vector< std::vector<double> >& data, std::vector<double>& meanMatrix);
void zscore(std::vector< std::vector<double> >& data, std::vector<double>& meanMatrix, std::vector<double>& stdMatrix);

void fileInput(const char dataset[], std::vector< std::vector<double> >& data, std::vector< std::vector<int> >& target);
void fileOutputWeights(const char fileName[], int cntFolds, std::vector<Neuron>);
void fileOutputConfusionMatrix(const char dataset[], int cntFolds, std::vector< std::vector<int> >,
							std::vector< std::vector<int> >, std::vector<double>, std::vector<double>, std::vector<double>,
							std::vector<double>, std::vector<double>, double, double, double, double, double);

void printFolds(std::vector< std::vector<double> >& trainingData, std::vector< std::vector<int> >& trainingTarget,
				std::vector< std::vector<double> >& validationData, std::vector< std::vector<int> >& validationTarget);

int main()
{
    /*
     *	1 - Perceptron Simples
     *	2 - Adaline
     */
    int idClassifier = 2;

    /*
     *	1 - Iris
     *	2 - Wine
     *	3 - Dermatology
     *	4 - Vertebral Column
     */
    int idDataset = 1;

    int k = 5; // qtd de folds de validacao.
	int numEpochs = 100;
	double learningRate = 0.1;
	double acceptableError = 0.01;

	std::vector< std::vector<double> > trainingData;
	std::vector< std::vector<int> > trainingTarget;

	std::vector< std::vector<double> > validationData;
	std::vector< std::vector<int> > validationTarget;

	std::vector<double> meanMatrix;
	std::vector<double> stdMatrix;

	switch(idDataset)
	{
		case 1:
			fileInput("datasets/iris.data", trainingData, trainingTarget);
			break;
		case 2:
			fileInput("datasets/wine.data", trainingData, trainingTarget);
			break;
		case 3:
			fileInput("datasets/dermatology.data", trainingData, trainingTarget);
			break;
		case 4:
			fileInput("datasets/column.data", trainingData, trainingTarget);
			break;
		default:
			cout << "Dataset nao especificado!" << endl;
			break;
	}

	// normalizacao z-score
    meanMatrix = mean(trainingData);
    stdMatrix = stdDeviation(trainingData, meanMatrix);
    zscore(trainingData, meanMatrix, stdMatrix);

    int numInputs = trainingData[0].size();
	int n = (trainingData.size() / k); // qtd de exemplos por fold
	validationData.resize(n); // matriz de teste eh, portanto, cada fold criado
	validationTarget.resize(n);

    cout << "Classificadores Lineares" << endl << endl;

    if(idClassifier == 1)
    {
		cout << "Perceptron Simples" << endl << endl;
		cout << " A G U A R D E   O   P R O C E S S A M E N T O" << endl;

		Perceptron perceptron;
		perceptron.config(1, numInputs, k);
		perceptron.trainingConfig(numEpochs, learningRate);

		int cnt = 0;
		for(int i = 0; i < k; i++)
		{
			cout << endl << "Fold-" << i << endl;

			// Constroi cada Fold de teste e treino.
			for(int j = cnt; j < (cnt + n); j++)
			{
				for(unsigned int l = 0; l < trainingData[j].size(); l++)
					validationData[(j % n)].push_back(trainingData[j][l]);

				for(unsigned int l = 0; l < trainingTarget[j].size(); l++)
					validationTarget[(j % n)].push_back(trainingTarget[j][l]);

				trainingData[j].clear();
				trainingTarget[j].clear();
			}

			perceptron.training(trainingData, trainingTarget);
			perceptron.operation(validationData, validationTarget);
			perceptron.computePerformanceMetrics(i, k);

			fileOutputWeights("weights.txt", i, perceptron.neurons);

			fileOutputConfusionMatrix("confusionMatrix.txt", i, perceptron.confusionMatrix, perceptron.confusionMatrixTotal,
					perceptron.accuracy, perceptron.errorRate, perceptron.precision, perceptron.sensitivity, perceptron.specificity,
					perceptron.meanAccuracy, perceptron.meanErrorRate, perceptron.meanPrecision, perceptron.meanSensivity, perceptron.meanSpecificity);

			for(int i = 0; i < 2; i++)
			{
				for(int j = 0; j < 2; j++)
					perceptron.confusionMatrix[i][j] = 0;
			}

			// Recompoe a matriz de treino e limpa a matriz de teste antes de construir um novo Fold.
		    for(int m = cnt; m < (cnt + n); m++)
		    {
		        for(unsigned int p = 0; p < validationData[(m % n)].size(); p++)
		        	trainingData[m].push_back(validationData[(m % n)][p]);

		        for(unsigned int p = 0; p < validationTarget[(m % n)].size(); p++)
		        	trainingTarget[m].push_back(validationTarget[(m % n)][p]);

		        validationData[(m % n)].clear();
		        validationTarget[(m % n)].clear();
		    }
			cnt = cnt + n;
		}
    }
    else if(idClassifier == 2)
    {
		cout << "Adaline" << endl << endl;
		cout << " A G U A R D E   O   P R O C E S S A M E N T O" << endl;

		Adaline adaline;
		adaline.config(1, numInputs, k);
		adaline.trainingConfig(numEpochs, learningRate, acceptableError);

		int cnt = 0;
		for(int i = 0; i < k; i++)
		{
			cout << endl << "Fold-" << i << endl;

			// Constroi cada Fold de teste e treino.
			for(int j = cnt; j < (cnt + n); j++)
			{
				for(unsigned int l = 0; l < trainingData[j].size(); l++)
					validationData[(j % n)].push_back(trainingData[j][l]);

				for(unsigned int l = 0; l < trainingTarget[j].size(); l++)
					validationTarget[(j % n)].push_back(trainingTarget[j][l]);

				trainingData[j].clear();
				trainingTarget[j].clear();
			}

			adaline.training(trainingData, trainingTarget);
			adaline.operation(validationData, validationTarget);
			adaline.computePerformanceMetrics(i, k);

			fileOutputWeights("weights.txt", i, adaline.neurons);

			fileOutputConfusionMatrix("confusionMatrix.txt", i, adaline.confusionMatrix, adaline.confusionMatrixTotal,
					adaline.accuracy, adaline.errorRate, adaline.precision, adaline.sensitivity, adaline.specificity,
					adaline.meanAccuracy, adaline.meanErrorRate, adaline.meanPrecision, adaline.meanSensivity, adaline.meanSpecificity);

			for(int i = 0; i < 2; i++)
			{
				for(int j = 0; j < 2; j++)
					adaline.confusionMatrix[i][j] = 0;
			}

			// Recompoe a matriz de treino e limpa a matriz de teste antes de construir um novo Fold.
		    for(int m = cnt; m < (cnt + n); m++)
		    {
		        for(unsigned int p = 0; p < validationData[(m % n)].size(); p++)
		        	trainingData[m].push_back(validationData[(m % n)][p]);

		        for(unsigned int p = 0; p < validationTarget[(m % n)].size(); p++)
		        	trainingTarget[m].push_back(validationTarget[(m % n)][p]);

		        validationData[(m % n)].clear();
		        validationTarget[(m % n)].clear();
		    }
			cnt = cnt + n;
		}
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

void fileOutputWeights(const char file[], int cntFolds, std::vector<Neuron> neurons)
{
    ofstream outFile(file, ios::app);

    if (!outFile)
    {
    	cerr << "O log dos pesos não pode ser criado!" << endl;
    	exit(1);
    }

    outFile << "\n" << "--------------------------------------------------" << "\n";
    outFile << "Fold-" << cntFolds << "\n";
	for(unsigned int n = 0; n < neurons.size(); n++)
	{
		for(unsigned int w = 0; w < neurons[n].weights.size(); w++)
			outFile << neurons[n].weights[w] << "\t";
		outFile << "\n";
	}
}

void fileOutputConfusionMatrix(const char dataset[], int cntFolds, std::vector< std::vector<int> > confusionMatrix,
		std::vector< std::vector<int> > confusionMatrixTotal, std::vector<double> acc, std::vector<double> err,
		std::vector<double> pre, std::vector<double> sen, std::vector<double> spe, double meanAcc, double meanErr,
		double meanPre, double meanSen, double meanSpe)
{
    ofstream outFile(dataset, ios::app);

    if (!outFile)
    {
    	cerr << "O log da matriz de confusao não pode ser criado!" << endl;
    	exit(1);
    }

    outFile << "\n" << "--------------------------------------------------" << "\n";
    outFile << "Fold-" << cntFolds << "\n";
    outFile << "\n" << "Matriz de Confusao" << "\n\n";
    outFile << "Real Predita" << "\n";
	for(unsigned int i = 0; i < confusionMatrix.size(); i++)
	{
		for(unsigned int j = 0; j < confusionMatrix[i].size(); j++)
			outFile << " [" << i << "]   [" << j << "]  =  " << confusionMatrix[i][j] << "\n";
		outFile << "\n";
	}

	outFile << "\n" << "Metricas de Desempenho" << "\n";
	outFile << "Accuracy = " << acc[cntFolds] << "\n";
	outFile << "Error Rate = " << err[cntFolds] << "\n";
	outFile << "Precision = " << pre[cntFolds] << "\n";
	outFile << "Sensitivity = " << sen[cntFolds] << "\n";
	outFile << "Specificity = " << spe[cntFolds] << "\n";

	if(cntFolds == 4)
	{
		outFile << "\n" << "--------------------------------------------------" << "\n";
	    outFile << "\n" << "Matriz de Confusao Total" << "\n\n";
	    outFile << "Real Predita" << "\n";
		for(unsigned int i = 0; i < confusionMatrixTotal.size(); i++)
		{
			for(unsigned int j = 0; j < confusionMatrixTotal[i].size(); j++)
				outFile << " [" << i << "]   [" << j << "]  =  " << confusionMatrixTotal[i][j] << "\n";
			outFile << "\n";
		}

		outFile << "\n" << "Metricas de Desempenho Medias" << "\n";
		outFile << "Accuracy Media = " << meanAcc << "\n";
		outFile << "Error Rate Media = " << meanErr << "\n";
		outFile << "Precision Media = " << meanPre << "\n";
		outFile << "Sensitivity Media = " << meanSen << "\n";
		outFile << "Specificity Media = " << meanSpe << "\n";
	}
}

void printFolds(std::vector< std::vector<double> >& trainingData, std::vector< std::vector<int> >& trainingTarget,
				std::vector< std::vector<double> >& validationData, std::vector< std::vector<int> >& validationTarget)
{
    for(unsigned int q = 0; q < trainingData.size(); q++)
    {
    	for(unsigned int u = 0; u < trainingData[q].size(); u++)
    		cout << "trainingData[" << q << "][" << u << "] = " << trainingData[q][u] << endl;

    	for(unsigned int u = 0; u < trainingTarget[q].size(); u++)
    		cout << "trainingTarget[" << q << "][" << u << "] = " << trainingTarget[q][u] << endl;
    }

    for(unsigned int v = 0; v < validationData.size(); v++)
    {
    	for(unsigned int x = 0; x < validationData[v].size(); x++)
    		cout << "validationData[" << v << "][" << x << "] = " << validationData[v][x] << endl;

    	for(unsigned int x = 0; x < validationTarget[v].size(); x++)
    		cout << "validationTarget[" << v << "][" << x << "] = " << validationTarget[v][x] << endl;
    }
}
