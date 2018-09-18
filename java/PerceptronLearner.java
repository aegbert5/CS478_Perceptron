// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import java.util.ArrayList;
import java.util.Random;

/**
 * For nominal labels, this model simply returns the majority class. For
 * continuous labels, it returns the mean value.
 * If the learning model you're using doesn't do as well as this one,
 * it's time to find a new learning model.
 */
public class PerceptronLearner extends SupervisedLearner {

	double learningRate = 0.05;
	Random rand;
	Perceptron[] perceptrons;

	public PerceptronLearner(Random rand) {
		this.rand = rand;
	}

	private int factorial(int value) {
		if (value < 2) {
			return 1;
		} else {
			return value * factorial(value - 1);
		}
	}

	private int calculateNumWeights(int numInputs, int order) {
		int cumulative = 0;

		for (int i = order; i > 0; --i) {
			cumulative += factorial(numInputs) / (factorial(numInputs - i) * factorial(i));
		}


		return cumulative;
	}

	// Preforms one epoch of training on the dataset
	public void train(Matrix features, Matrix labels) throws Exception {

		// Check how many perceptrons need to be created for a classification system
		int numClasses = labels.valueCount(0);
		perceptrons = new Perceptron[numClasses];

		int numInputs = features.row(0).length;

		int numWeights;
		if (quadratic) {
			numWeights = calculateNumWeights(numInputs, 2) + 1;
		} else {
			numWeights = numInputs + 1;
		} 

		int trainingLength = features.rows();

		// If the output is a continuous ranking
		if (numClasses > 1) {

			for (int i = 0; i < numClasses; ++i) {
				perceptrons[i] = new Perceptron(rand, learningRate, numWeights, i, labels.attrValue(0, i));
			}

			System.out.println();
			System.out.println("Perceptron training in progress...");
			System.out.println("# of Perceptrons: " + numClasses);
			System.out.println("| Inputs (w/ Bias) |   Weights   | Target | Net |  ^Weights   |");

			double previousWeightChange = Double.MAX_VALUE;
			double weightChange = Double.MAX_VALUE;
			double minimumChangeThreshold = learningRate * 1.0; 
			int totalEpochs = 0;
			int minNumEpochs = 5;

			// Execute until change is less than threshold
			do {
				System.out.println("Training Epoch: " + ++totalEpochs);
				features.shuffle(rand, labels);

				previousWeightChange = weightChange;
				weightChange = 0.0;

				for (int i = 0; i < trainingLength; ++i) {
					double net = 0;
					double[] currentData = new double[numWeights]; 

					// Initialize the first data to be equivalent to the features
					for (int j = 0; j < features.row(i).length; ++j) {
						currentData[j] = features.row(i)[j];
					}

					// Then add the second order data
					for (int j = features.row(i).length, a = 0; a < features.row(i).length; ++a) {
						for (int b = a+1; b < features.row(i).length; ++b) {
							currentData[j] = features.row(i)[a] * features.row(i)[b];
							++j;
						}
					}

					for (int j = 0; j < numClasses; ++j) {
						perceptrons[j].train(currentData, labels.get(i, 0));
					}
					
				}
				// System.out.println("---------------------------");
				// System.out.println("Threshold: " + minimumChangeThreshold);
				// System.out.println("Total Weight Change: " + weightChange);
				// System.out.println("Previous Weight Change: " + previousWeightChange);
				// System.out.println("---------------------------");
			} while (totalEpochs < minNumEpochs); //&& weightChange > minimumChangeThreshold && weightChange < previousWeightChange);
		} else { // Else, the output if continuous 
			System.out.println("Perceptron training does not handle continuous ranking learning. NO TRAINING OCCURRED");
		}
	}

	// private void printPerceptrons(double[] currentData, Perceptron[] perceptrons) {

	// 	System.out.print("| ");
	// 	for (int j = 0; j < currentData.length; ++j) {
	// 		System.out.print(currentData[j] + ", ");
	// 	}
	// 	System.out.println("1.0 ");

	// 	// Debug
	// 	for (int j = 0; j < numClasses; ++j) {
	// 		perceptrons[j].printWeights();
	// 	}
	// }

	public void predict(double[] features, double[] labels) throws Exception {
		
		double highestNet = 0.0;
		int highestPerceptronIndex = 0;

		int numPerceptrons = perceptrons.length;

		for (int i = 0; i < numPerceptrons; ++i) {
			double currentNet = perceptrons[i].predict(features);
			if (currentNet > highestNet) {
				highestNet = currentNet;
				highestPerceptronIndex = i;
			}
		}

		labels[0] = highestPerceptronIndex;
	}

}
