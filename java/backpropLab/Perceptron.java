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
public class Perceptron {

	Random rand;
	double learningRate;
	double[] weights;
	int numWeights;

	int classNumber; // Ranges from 1 to n
	String className;

  // Constants
	boolean debug = false;
  boolean printWeights = false;

	public Perceptron(Random rand, double learningRate, int numWeights, int classNumber, String className) {
		this.rand = rand;
		this.learningRate = learningRate;
		this.weights = new double[numWeights];
    this.numWeights = numWeights;
		this.classNumber = classNumber;
		this.className = className;

		//Initialize weights to random values between -1 and 1
		for (int i = 0; i < numWeights; ++i) {
			this.weights[i] = -1.0 + rand.nextInt(2);
		}
	}

	// Preforms one epoch of training on the dataset
	public void train(double[] currentData, double targetClassNumber) throws Exception {

		double net = 0;

		for (int j = 0; j < currentData.length; ++j) {
			net += currentData[j] * weights[j];
		}

		// Then add the bias into the perceptron
		net += weights[numWeights - 1];

		double currentTarget;
		double output;

		if (targetClassNumber != classNumber) {
			currentTarget = 0.0;
		} else {
			currentTarget = 1.0;
		}

		if (net > 0) {
			output = 1.0;
		} else {
			output = 0.0;
		}

		double weightChange = 0.0;

		if (debug) {
			System.out.print("Perceptron " + classNumber + " (" + className + "): | ");

			for (int j = 0; j < currentData.length; ++j) {
				System.out.print(currentData[j] + ", ");
			}
		  System.out.print("| ");

			for (int j = 0; j < weights.length; ++j) {
				System.out.print(weights[j] + ", ");
			}

			System.out.print("| " + currentTarget + " | " + output + " | ");
		}

    // Change the weights if the output and current tager do not match
		if (currentTarget != output) {
			for (int j = 0; j < currentData.length; ++j) {
				double change = learningRate * (currentTarget - output) * currentData[j];
				weightChange += Math.abs(change);
				if (debug) {
					System.out.print(change + ", ");
				}
				weights[j] += change;
			}

			double biasChange = learningRate * (currentTarget - output);
			if (debug) {
				System.out.print(biasChange + ", ");
			}
			// Change the bias weight
			weightChange += Math.abs(biasChange);
			weights[weights.length - 1] = biasChange;
		} else {
			// DONT Change the weights
			if (debug) {
				System.out.print("No Change ");
			}
		}

		if (debug) {
			System.out.println("| Total Weight Change: " + weightChange);
		}

    if (printWeights) {
      System.out.print("Perceptron " + classNumber + " (" + className + ") weights: | ");

			for (int j = 0; j < currentData.length; ++j) {
				System.out.print(weights[j] + ", ");
			}

      System.out.println();
    }
	}

	public double predict(double[] features) {
		
		double net = 0.0;

		for (int i = 0; i < features.length; ++i) {
			net += features[i] * weights[i];
		}

		// Then add the bias into the perceptron
		net += weights[numWeights - 1];
		return net;
	}

}
