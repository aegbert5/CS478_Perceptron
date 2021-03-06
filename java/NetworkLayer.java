import java.util.Random;
import java.lang.Math;

public class NetworkLayer {

  double[][] weights;
  double[][] deltaWeights;
  double[] outputData;
  double[] inputData;
  int numRows;
  int numColumns;

  boolean isMomentumChange = true;
  double momentumValue = 0.9;

  double learningRate = 0.1;
  Random rand;
  int layerIndex;

  boolean debug = false;
  
  public NetworkLayer(Random rand, int layerIndex, int numNodes, int numNextNodes) {

    // Add one to the weight matrix to signify the bias node
    ++numNodes;

    this.weights = new double[numNodes][numNextNodes];

    // Initialize weights to random values between -1 and 1 
    for (int i = 0; i < numNodes; ++i) {
      for (int j = 0; j < numNextNodes; ++j) {
        this.weights[i][j] = (rand.nextDouble() * 2) - 1;
      }
    }

    this.deltaWeights = new double[numNodes][numNextNodes];
    this.outputData = new double[numNextNodes + 1]; // to insert the bias value
    this.numRows = numNodes;
    this.numColumns = numNextNodes;
    this.learningRate = learningRate;
    this.rand = rand;
    this.layerIndex = layerIndex;
  }

  // Input Data is multiplied by the weight matrix to get the resulting output values
  // This includes applying the output values
  public double[] getOutputData(double[] inputData) throws Exception {
     int inputSize = inputData.length;

     printArray("LAYER INPUT (" + layerIndex + "):", inputData);
     printMatrix("WEIGHTS:", weights);

     this.inputData = inputData;
     
     if (inputSize != numRows) {
      throw new Exception("Input size invalid for OUTPUT DATA: (numInputs, numRows) -> (" + inputSize + ", " + numRows + ") on layer index " + layerIndex);
     } else {
      //  Be careful not to modify the outputData after the return statement, since this value is needed for further computation in the object

        for (int j = 0; j < numColumns; ++j) {
          double sum = 0;
          for (int i = 0; i < numRows; ++i) {
            sum += inputData[i] * weights[i][j];
          }
          outputData[j] = getActivationValue(sum);
        }

        // Add the bias value to the output
        outputData[numColumns] = 1;

        return outputData;
     }
  }

  public double[] updateWeightsAndGetDeltaData(double[] currentDeltaData) throws Exception {

    int inputSize = currentDeltaData.length;
      
    printArray("LAYER DELTA (" + layerIndex + "):", currentDeltaData);

    if (inputSize != numColumns) {
      System.out.println("ERROR ---------------");
      printMatrix("NETWORK WEIGHTS:", weights);
      throw new Exception("Input size invalid for DELTA DATA: (numInputs, numColumns) -> (" + inputSize + ", " + numColumns + ") on layer index " + layerIndex);
    } else {
      double[] newDeltaData = getDeltaData(currentDeltaData);

      // Update weights always after delta data has been retreived
      updateWeights(currentDeltaData);
      return newDeltaData;
    }
  }

  private void updateWeights(double[] currentDeltaData) {
    // This does not do matrix multiplication, but instead creates a delta weight matrix
    // This delta weight matrix allows for momentum terms to be optionally included
    // This will ignore the bias data from the output
    for (int j = 0; j < numColumns; ++j) {

      double currentError = currentDeltaData[j];


      int numOutputData = outputData.length;

      for (int i = 0; i < numRows; ++i) {
        // (i,j) -> the ith output node to the next jth input node of the next layer
        double delta = learningRate * inputData[i] * currentError;

        // If the momentum term is applied, then use the deltaWeights table to get the actual weight change
        if (isMomentumChange) {
          double momentumDelta = momentumValue * deltaWeights[i][j];
          weights[i][j] += delta + momentumDelta;
          deltaWeights[i][j] = delta + momentumDelta;
        } else {
          weights[i][j] += delta;
          deltaWeights[i][j] = delta; // For records
        }
      }
    }
    
    printMatrix("NEW WEIGHTS:", weights);
  }

  // Data is of correct size at this point
  private double[] getDeltaData(double[] currentDeltaData) {
    double[] nextDelta = new double[numRows -1];

    int inputSize = currentDeltaData.length; // equivalent to numRows

    // This will ignore the bias data from the input
    for (int i = 0; i < numRows - 1; ++i) {
      double sumOfWeightsAndDeltas = 0;
     
      double fNet = getFNet(inputData[i]);
      for (int j = 0; j < inputSize; ++j) {
        sumOfWeightsAndDeltas += currentDeltaData[j] 
          * weights[i][j];
      }

      nextDelta[i] = sumOfWeightsAndDeltas * fNet;
    }

    return nextDelta;
  }

  private double getFNet(double outputValue) {
    return outputValue * (1 - outputValue);
  }

  private double getActivationValue(double netSum) {
    return 1.0 / (1.0 + Math.pow(Math.E, -netSum));
  }

  private void printMatrix(String label, double[][] matrix) {
    if (debug) {
    System.out.println(label);
    System.out.println("[");
    for (int i = 0; i < numRows; ++i) {
      System.out.print("[");
      for (int j = 0; j < numColumns; ++j) {
        System.out.printf("%.6f", matrix[i][j]);
        if (j != numColumns - 1) {
          System.out.print(", ");
        }
      }
      System.out.println("]");
    }
    System.out.println("]");
    }
  }

  private void printArray(String label, double[] inputValues) {
    if (debug) {
    System.out.println(label);
    System.out.print("[");
    int length = inputValues.length;
    for (int i = 0; i < length; ++i) {
      System.out.printf("%.6f",inputValues[i]);
      if (i != length - 1) {
        System.out.print(", ");
      }
    }
    System.out.println("]");
    }
  }
}
