import java.util.ArrayList;
import java.util.Random;
import java.util.List;
import java.io.PrintWriter;
import java.io.FileWriter;
import java.io.IOException;

public class NeuralNet extends SupervisedLearner {

  int numHiddenLayers = 1;
  int numTotalLayers = numHiddenLayers + 1; // 1 additional for input layer
  int[] numPerceptronsPerLayer = new int[]{4}; // Excluding bias nodes, value > 0
  NetworkLayer[] networkLayers = new NetworkLayer[numTotalLayers];

  int numInitialWeights;
  boolean isContinuous;
  int numOutputs; // if continuous -> 1, else numClasses

  int minEpochs = 1000;
  int epochWindow = 10; // How many epochs will be run before checking stopping criteria again
  double minAccuracyImprovement = 0.0001;
  long maxRuntimeSeconds = 30; // Max seconds in which the neural net will run

  Matrix validationFeatures;
  Matrix validationLabels;
  Random rand;

  boolean debug = false;

  public NeuralNet(Random rand) {
    this.rand = rand;
  }

  @Override
  public void setValidationSet(Matrix validationFeatures, Matrix validationLabels) {
    this.validationFeatures = validationFeatures;
    this.validationLabels = validationLabels;
  }

	public void train(Matrix features, Matrix labels) throws Exception {

    // Set validation set for stopping criteria if not set already
    if (validationFeatures == null) {
      validationFeatures = features;
    }

    if (validationLabels == null) {
      validationLabels = labels;
    }

    long startMilliseconds = System.currentTimeMillis();

    int numInputs = features.row(0).length;
    int numClasses = labels.valueCount(0);

    numInitialWeights = numInputs + 1;

    int trainingLength = features.rows();

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

    int epochCount = 0;
    double currentAccuracy;
    double bestAccuracy;// = Integer.MIN_VALUE;
    double tsAccuracy = 0.0;

    if (isContinuous) {
      currentAccuracy = Integer.MAX_VALUE;
      bestAccuracy = Integer.MAX_VALUE;
    } else {
      currentAccuracy = Integer.MIN_VALUE;
      bestAccuracy = Integer.MIN_VALUE;
    }

    List<String> gifImages = new ArrayList<String>();

    do {

      if (epochCount % epochWindow == 0) {
        if ((isContinuous && currentAccuracy < bestAccuracy) || (!isContinuous && currentAccuracy > bestAccuracy)) {
          bestAccuracy = currentAccuracy;
        }
      }

      features.shuffle(rand, labels);
        
      if (debug) {
        System.out.println("---------------------Epoch " + epochCount + "------------------");
      }
        
      for (int i = 0; i < trainingLength; ++i) {
        double[] currentData = new double[numInitialWeights];

        // Initialize the current data to be equivalent to the features
        for (int j = 0; j < numInitialWeights - 1; ++j) {
          currentData[j] = features.row(i)[j];
        }

        // Add the bias input value
        currentData[numInitialWeights - 1] = 1;

        if (debug) {
          System.out.println("-------------------------- NEW TRAINING SET (" + (i + 1) + " of " + trainingLength + ") -> Forward propagating... ----------------------");
        }

        trainNetwork(currentData, labels.get(i, 0), isContinuous);
      }

      validationFeatures.shuffle(rand, validationLabels);
      currentAccuracy = measureAccuracy(validationFeatures, validationLabels, null);
      tsAccuracy = measureAccuracy(features, labels, null);

      ++epochCount;

      // Accuracy values are MSE with continuous data, % average accuracy when nominal output

      // Used for exporting to gif animation
      //if (epochCount % epochWindow == 0) {
        //gifImages.add(exportAccuracy(epochCount));
      //}
    } while ((epochCount < minEpochs || isImprovedOverWindow(currentAccuracy, bestAccuracy, minAccuracyImprovement, epochCount)) && System.currentTimeMillis() - startMilliseconds < maxRuntimeSeconds * 1000);

    // Training is done, print out accuracy results
    //gifImages.add(exportAccuracy(epochCount));
    //GifSequenceWriter.generateGif(gifImages, "images/accuracyGif.gif");
    
  }

  private boolean isImprovedOverWindow(double currentAccuracy, double bestAccuracy, double minAccuracyImprovement, int epochCount) {
    if (epochCount % epochWindow == 0) {
      if ((isContinuous && currentAccuracy < bestAccuracy - minAccuracyImprovement) || (!isContinuous && currentAccuracy > bestAccuracy + minAccuracyImprovement)) {
        return true;
      } else {
        return false;
      }
    } else {
      return true;
    }
  }

  private double trainNetwork(double[] currentInputData, double targetOutput, boolean isContinuous) throws Exception {

    printArray("FEATURE DATA:", currentInputData);
    
    // Mutlitply the current node values by the weight matrix, and apply the output function to the sums internally
    for (int i = 0; i < numTotalLayers; ++i) {
      NetworkLayer currentLayer = networkLayers[i];
      double[] nextOutputData = currentLayer.getOutputData(currentInputData);
      currentInputData = nextOutputData;
    }

    printArray("OUTPUT DATA:", currentInputData);
    if (debug) {
      System.out.println("->->->->->->->->->-> Back propagating... <-<-<-<-<-<-<-<-<-<-<-<-<-<-");
    }

    // Subtract 1 to ignore the bias added to the outputData
    int outputDataLength = currentInputData.length - 1;
    
    double[] targetValues = new double[outputDataLength];
    double[] currentDeltaData = new double[outputDataLength];

    if (isContinuous) {
      if (debug) {
        System.out.println("TARGET VALUE: " + targetOutput);
      }
      targetValues[0] = targetOutput;
    } else {
      if (debug) {
        System.out.println("TARGET INDEX: " + (int) targetOutput);
      }
      targetValues[(int) targetOutput] = 1;
    }

    // Gets the backpropogation error from the final output data
    double sumSquaredError = getSSEAndDeltaData(targetValues, currentInputData, currentDeltaData);
    
    // Take the last layer's delta data to calculate the delta data for the next layer to use 
    // Each layer will update their weights accordingly at this point
    for (int i = numTotalLayers - 1; i >= 0; --i) {
      NetworkLayer currentLayer = networkLayers[i];
      double[] nextDeltaData = currentLayer.updateWeightsAndGetDeltaData(currentDeltaData);
      currentDeltaData = nextDeltaData;
    }

    return sumSquaredError;
  }

  private double getSSEAndDeltaData(double[] targetValues, double[] outputData, double[] deltaData) {

    int outputDataLength = targetValues.length;
    
    double sumSquaredError = 0;
    
    for (int i = 0; i < outputDataLength; ++i) {
      double currentOutputData = outputData[i];
      double currentTarget = targetValues[i];
      deltaData[i] = (currentTarget - currentOutputData) * getFNet(currentOutputData);
      sumSquaredError += Math.pow(currentTarget - currentOutputData, 2);
    }
    
    return sumSquaredError;
  }

  private void printArray(String label, double[] array) {
    if (debug) {
    System.out.print(label);
    System.out.print(" [");
    int length = array.length;
    for (int i = 0; i < length; ++i) {
      System.out.printf("%.3f", array[i]);
      if (i != length - 1) {
        System.out.print(", ");
      }
    }
    System.out.println("]");
    }
  }

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

      //Pick the class number that corresponds to the highest output value
      for (int i = 0; i < numOutputs; ++i) {
        double currentOutput = currentInputData[i];
        if (currentOutput > highestNet) {
          highestNet = currentOutput;
          highestNetClassNumber = i;
        }
      }

      labels[0] = highestNetClassNumber;
    }
  }

  // Returns the filename of the ppm image generated
  private String exportAccuracy(int epochNumber) throws Exception {

    // Lower bounds
    double startX = -1.0;
    double startY = -1.0;

    // Upper bounds
    double endX = 1.0;
    double endY = 1.0;

    // Deltas
    double xDelta = 0.01;
    double yDelta = 0.01;

    String imageFileName = "images/accuracy" + epochNumber + ".ppm";
    FileWriter fileWriter = new FileWriter(imageFileName);

    PrintWriter writer = new PrintWriter(fileWriter);

    //writer.printf("P3\n%d %d\n255\n", (int) ((endX - startX) / xDelta), (int) ((endY - startY) / yDelta));

    // Starts from top-left to bottom-right
    for (double y = endY; y > startY; y -= yDelta) {
      for (double x = startX; x < endX; x += xDelta) {
        double[] features = new double[]{x, y};
        double[] labels = new double[1];
        predict(features, labels);
        int prediction = ((int) (labels[0] * 255));
        if (!isContinuous) {
          prediction /= (numOutputs - 1);
        }
        writer.printf("%d %d %d ", prediction, prediction, prediction);
      }
      writer.printf("\n");
    }

    writer.close();

    return imageFileName;
  }
}
