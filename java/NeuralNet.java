import java.util.ArrayList;
import java.util.Random;
import java.io.PrintWriter;
import java.io.FileWriter;
import java.io.IOException;

public class NeuralNet extends SupervisedLearner {

  int numHiddenLayers = 2;
  int numTotalLayers = numHiddenLayers + 1; // 1 additional for input layer
  int[] numPerceptronsPerLayer = new int[]{4, 4}; // Excluding bias nodes, value > 0
  NetworkLayer[] networkLayers = new NetworkLayer[numTotalLayers];

  int numInitialWeights;
  boolean isContinuous;

  Random rand;

  boolean debug = false;

  public NeuralNet(Random rand) {
    this.rand = rand;
  }

	public void train(Matrix features, Matrix labels) throws Exception {

    int numInputs = features.row(0).length;
    int numClasses = labels.valueCount(0);

    numInitialWeights = numInputs + 1;

    int trainingLength = features.rows();

    int numOutputs;

    if (numClasses > 1) {
      isContinuous = false;
      numOutputs = numClasses;
    } else {
      // If continuous, then there should only be one output of data
      isContinuous = true;
      numOutputs = 1;
    }
      
    int numCurrentNodes = 0;
    int numNextNodes = numInputs; // No qudratic outputs

      // Create the necessary hidden layers for the network
      for (int i = 0; i < numTotalLayers; ++i) {
        numCurrentNodes = numNextNodes;
        if (i == numTotalLayers - 1) {
          numNextNodes = numOutputs;
        } else {
          numNextNodes = numPerceptronsPerLayer[i];
        }
        networkLayers[i] = new NetworkLayer(rand, i, numCurrentNodes, numNextNodes);
      }

      int epochs = 0;

      do {

        features.shuffle(rand, labels);
        
        //System.out.println("---------------------Epoch " + epochs + "------------------");

        for (int i = 0; i < trainingLength; ++i) {
          double[] currentData = new double[numInitialWeights];

          // Initialize the current data to be equivalent to the features
          for (int j = 0; j < numInitialWeights - 1; ++j) {
            currentData[j] = features.row(i)[j];
          }

          // Add the bias input value
          currentData[numInitialWeights - 1] = 1;

          //System.out.println("-------------------------- NEW TRAINING SET (" + (i + 1) + " of " + trainingLength + ") -> Forward propagating... ----------------------");

          trainNetwork(currentData, labels.get(i, 0), isContinuous);
        }

        ++epochs;
      } while (epochs < 9000);

      // Training is done, print out accuracy results
      exportAccuracy();
  }

  private void trainNetwork(double[] currentInputData, double targetOutput, boolean isContinuous) throws Exception {

    printArray("FEATURE DATA:", currentInputData);
    
    // Mutlitply the current node values by the weight matrix, and apply the output function to the sums internally
    for (int i = 0; i < numTotalLayers; ++i) {
      NetworkLayer currentLayer = networkLayers[i];
      double[] nextOutputData = currentLayer.getOutputData(currentInputData);
      currentInputData = nextOutputData;
    }

    printArray("OUTPUT DATA:", currentInputData);
    //System.out.println("->->->->->->->->->-> Back propagating... <-<-<-<-<-<-<-<-<-<-<-<-<-<-");

    // Gets the backpropogation error from the final output data
    double[] currentDeltaData = getDeltaData(currentInputData, targetOutput, isContinuous);

    // Take the last layer's delta data to calculate the delta data for the next layer to use 
    // Each layer will update their weights accordingly at this point
    for (int i = numTotalLayers - 1; i >= 0; --i) {
      NetworkLayer currentLayer = networkLayers[i];
      double[] nextDeltaData = currentLayer.updateWeightsAndGetDeltaData(currentDeltaData);
      currentDeltaData = nextDeltaData;
    }

  }

  private double[] getDeltaData(double[] outputData, double targetOutput, boolean isContinuous) {

    // Subtract 1 to ignore the bias added to the outputData
    int outputDataLength = outputData.length - 1;
    
    //System.out.println("OUTPUT SIZE: " + outputDataLength);

    double[] targetValues = new double[outputDataLength];
    double[] deltaData = new double[outputDataLength];

    //System.out.println(targetClassNumber);
    if (isContinuous) {
      targetValues[0] = targetOutput;
    } else {
      targetValues[(int) targetOutput] = 1;
    }
    //printOutput(targetValues, outputData, outputDataLength);
    
    //targetValues[targetClassNumber] = 1;
    
    for (int i = 0; i < outputDataLength; ++i) {
      double currentOutputData = outputData[i];
      deltaData[i] = (targetValues[i] - currentOutputData) * getFNet(currentOutputData);
    }
    

    return deltaData;
  }

  private void printArray(String label, double[] array) {
    if (debug) {
    System.out.print(label);
    System.out.print(" [");
    int length = array.length;
    for (int i = 0; i < length; ++i) {
      System.out.print(array[i]);
      if (i != length - 1) {
        System.out.print(", ");
      }
    }
    System.out.println("]");
    }
  }

  // TODO: Function repeated in network class, remove one of them
  private double getFNet(double outputValue) {
    return outputValue * (1 - outputValue);
  }

  public void predict(double[] features, double[] labels) throws Exception {

    double[] currentInputData = new double[numInitialWeights];

    for (int i = 0; i < features.length; ++i) {
      currentInputData[i] = features[i];
    }

    // Add the bias input
    currentInputData[numInitialWeights - 1] = 1;

    for (int i = 0; i < numTotalLayers; ++i) {
      NetworkLayer currentLayer = networkLayers[i];
      double[] nextOutputData = currentLayer.getOutputData(currentInputData);
      currentInputData = nextOutputData;
    }

    if (isContinuous) {
      labels[0] = currentInputData[0];
    } else {

      double highestNet = Integer.MIN_VALUE;
      int highestNetClassNumber = 0;

      // Get ride of the bias added to the end of the array
      int outputDataLength = currentInputData.length - 1;

      //Pick the class number that corresponds to the highest output value
      for (int i = 0; i < outputDataLength; ++i) {
        double currentOutput = currentInputData[i];
        if (currentOutput > highestNet) {
          highestNet = currentOutput;
          highestNetClassNumber = i;
        }
      }

      //printArray("HIGHEST FOUND:", currentInputData, outputDataLength);
      //System.out.println("HIGHEST: " + highestNetClassNumber);
      labels[0] = highestNetClassNumber;
    }
  }

  private void exportAccuracy() throws Exception {

    // Lower bounds
    double startX = -1.0;
    double startY = -1.0;

    // Upper bounds
    double endX = 1.0;
    double endY = 1.0;

    // Deltas
    double xDelta = 0.01;
    double yDelta = 0.01;

    String imageFileName = "accuracy.ppm";
    FileWriter fileWriter = new FileWriter(imageFileName);

    PrintWriter writer = new PrintWriter(fileWriter);

    writer.printf("P3\n%d %d\n255\n", (int) ((endX - startX) / xDelta), (int) ((endY - startY) / yDelta));

    // Starts from top-left to bottom-right
    for (double y = endY; y > startY; y -= yDelta) {
      for (double x = startX; x < endX; x += xDelta) {
        double[] features = new double[]{x, y};
        double[] labels = new double[1];
        predict(features, labels);
        int prediction = (int) (labels[0] * 255);
        writer.printf("%d %d %d ", prediction, prediction, prediction);
      }
      writer.printf("\n");
    }

    writer.close();
  }
}
