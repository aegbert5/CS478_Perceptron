import java.util.ArrayList;
import java.util.Random;

public class PerceptronLearner extends SupervisedLearner {

	Random rand;
	Perceptron[] perceptrons;
  int numWeights = 0;
  
  // Constants
  boolean quadratic = true;
  double learningRate = 0.1;
	int minNumEpochs = 10;
  double minimumAccuracyImprovement = 0.01; 
  
  public PerceptronLearner(Random rand) {
		this.rand = rand;
	}

	private int summate(int value) {
		if (value <= 1) {
			return 1;
		} else {
			return value + summate(value - 1);
		}
	}

	private int calculateNumWeightsForQuadratic(int numInputs) {
    return numInputs + summate(numInputs);
  }

	// Preforms one epoch of training on the dataset
	public void train(Matrix features, Matrix labels) throws Exception {

    // Check how many perceptrons need to be created for a classification system
		int numClasses = labels.valueCount(0);
		perceptrons = new Perceptron[numClasses];

		int numInputs = features.row(0).length;

		if (quadratic) {
			numWeights = calculateNumWeightsForQuadratic(numInputs) + 1;
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
      System.out.println("# of weights: " + numWeights);
			System.out.println("| Inputs (w/ Bias) |   Weights   | Target | Net |  ^Weights   |");

      System.out.println("Initial Accuracy: " + measureAccuracy(features, labels, null));

			double previousWeightChange = Double.MAX_VALUE;
			double weightChange = Double.MAX_VALUE;
			int totalEpochs = 0;

      double secondPreviousAccuracy = 0.0;
      double previousAccuracy = 0.0;
      double currentAccuracy = 0.0;

			// Execute until change is less than threshold
			do {

        secondPreviousAccuracy = previousAccuracy;
        previousAccuracy = currentAccuracy;

        ++totalEpochs;
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
          if (quadratic) {
			  		for (int j = features.row(i).length, a = 0; a < features.row(i).length; ++a) {
		  				for (int b = a; b < features.row(i).length; ++b) {
  							currentData[j] = features.row(i)[a] *
                  features.row(i)[b];
							  ++j;
						  }
					  }
          }

          // Add the bias
          currentData[currentData.length - 1] = 1.0;

					for (int j = 0; j < numClasses; ++j) {
						perceptrons[j].train(currentData, labels.get(i, 0));
					}
					
				}

        currentAccuracy = measureAccuracy(features, labels, null);

        System.out.println("---------------------------");
			  System.out.println("Training Epoch: " + totalEpochs);
				System.out.println("Accurracy: " + currentAccuracy);
			} while (totalEpochs < minNumEpochs
          || (totalEpochs >= minNumEpochs && (currentAccuracy >= previousAccuracy + minimumAccuracyImprovement))
          || (totalEpochs >= minNumEpochs && (currentAccuracy < previousAccuracy || currentAccuracy < secondPreviousAccuracy)));
		} else { // Else, if the data is continuous 
			System.out.println("Perceptron training does not handle continuous ranking learning. NO TRAINING OCCURRED");
		}
	}

	public void predict(double[] features, double[] labels) throws Exception {
		
		double highestNet = 0.0;
		int highestPerceptronIndex = 0;

		int numPerceptrons = perceptrons.length;

		for (int i = 0; i < numPerceptrons; ++i) {
			
      double[] currentData = new double[numWeights]; 

			// Initialize the first data to be equivalent to the features
			for (int j = 0; j < features.length; ++j) {
				currentData[j] = features[j];
			}

			// Then add the second order data
      if (quadratic) {
	  		for (int j = features.length, a = 0; a < features.length; ++a) {
  				for (int b = a; b < features.length; ++b) {
            //System.out.println("j => " + j + " -> (A,B) is: (" + a + "," + b + ")");
			  		currentData[j] = features[a] * features[b];
		  			++j;
	  			}
  			}
      }
      
      double currentNet = perceptrons[i].predict(currentData);
			if (currentNet > highestNet) {
				highestNet = currentNet;
				highestPerceptronIndex = i;
			}
		}

		labels[0] = highestPerceptronIndex;
	}

}
