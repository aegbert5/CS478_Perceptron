import java.util.*;
import javafx.util.Pair;

class InstanceBasedLearner extends SupervisedLearner {
  
  Random rand;
  boolean distanceWeighting = true;
  int k = 11;
  int numOutputClasses = 0;

  Matrix trainingFeatures;
  Matrix trainingLabels;
  int numTrainingRows;
  int numFeatures;
  boolean[] isContinuous; // Stores whether a feature at that index is continuous or nominal data
  double trainingSetPercentage = 1.0; // Percentage of the training set to keep (out of 1.0)

  public InstanceBasedLearner(Random rand) {
    this.rand = rand;
  }

  public void train(Matrix features, Matrix labels) throws Exception {

    numTrainingRows = (int) Math.floor(features.rows() * trainingSetPercentage);
    numFeatures = features.row(0).length;
    numOutputClasses = labels.valueCount(0);
   
    features.shuffle(rand, labels);

    trainingFeatures = new Matrix(features, 0, 0, numTrainingRows, numFeatures);
    trainingLabels = new Matrix(labels, 0, 0, numTrainingRows, 1); 

    isContinuous = new boolean[numFeatures];

    for (int i = 0; i < numFeatures; ++i) {
      isContinuous[i] = features.valueCount(i) < 2;
    }

  }

  private double getDistance(double[] features, double[] trainingInstance) {
    
    double distance = 0.0;
    
    for (int i = 0; i < numFeatures; ++i) {
      double difference = 0.0;
      if (isContinuous[i]) {
        difference = Math.abs(features[i] - trainingInstance[i]);
      } else {
        if (features[i] == Double.MAX_VALUE) {
          difference = 1.0;
        } else {
          if (features[i] == trainingInstance[i]) {
            difference = 0.0;
          } else {
            difference = 1.0;
          }
        }
      }
      distance += difference;
    }

    return distance;
  }

  public class ClosestComparator implements Comparator {
    public int compare(Object o1, Object o2) {

      Pair<Integer, Double> p1 = (Pair<Integer, Double>) o1;
      Pair<Integer, Double> p2 = (Pair<Integer, Double>) o2;

      double firstDist = p1.getValue();
      double secondDist = p2.getValue();
      
      if (firstDist > secondDist) {
        return -1;
      } else {
        return 1;
      }
    }
  }

  public void predict(double[] features, double[] labels) throws Exception {

    PriorityQueue<Pair<Integer, Double>> indexesOfClosest = new PriorityQueue(k, new ClosestComparator());

    for (int i = 0; i < numTrainingRows; ++i) {
      double[] currentRow = trainingFeatures.row(i);
      double distanceToInstance = getDistance(features, currentRow);
      if (indexesOfClosest.size() < k) {
        indexesOfClosest.add(new Pair(i, distanceToInstance));
      } else {
        if (distanceToInstance < indexesOfClosest.peek().getValue()) {
          indexesOfClosest.poll();
          indexesOfClosest.add(new Pair(i, distanceToInstance));
        }
      }
    }

    // Used for continuous data
    double continuousOutput = 0.0;
    double continuousDenominator = 0.0;

    Iterator<Pair<Integer, Double>> queueIterator = indexesOfClosest.iterator();
    Map<Double, Double> outputCount = new HashMap();

    // Count up the times an output appears in the closest neighbors
    while (queueIterator.hasNext()) {
      Pair<Integer, Double> pair = queueIterator.next();
      int currentIndex = pair.getKey();
      //System.out.print(currentIndex + ", ");
      double currentDistance = pair.getValue();

      double[] currentRow = trainingFeatures.row(currentIndex);
      double currentOutput = trainingLabels.row(currentIndex)[0];

      if (currentDistance == 0) {
        labels[0] = currentOutput;
        return;
      }

      // If classification output
      if (numOutputClasses > 1) {
        double currentOutputValue = outputCount.getOrDefault(currentOutput, 0.0);

        if (distanceWeighting) { // currentOutputValue represents the distance weighting
          currentOutputValue += 1 / (Math.pow(currentDistance, 2));
          outputCount.put(currentOutput, currentOutputValue); 
        } else { // it represents the number of occurances
          outputCount.put(currentOutput, ++currentOutputValue);
        }
      } else {
       // Do regression with continuous output, whether weighted or not
        if (distanceWeighting) {
          double tempValue = 1 / (Math.pow(currentDistance, 2));

          continuousOutput += tempValue * currentOutput;
          continuousDenominator += tempValue;

        } else {
          // Average of outputs
          continuousOutput += currentOutput / k;
        }
      }
    }

    // Finalize output class 
    if (numOutputClasses > 1) {
      double bestOutputValue = -1.0;
      double valueToBeat = -1.0;
      boolean valueChanged = false;
      
      for (Double outputClass : outputCount.keySet()) {
        double currentOutputValue = outputCount.get(outputClass);
        // currentOutputValue respresents the distance weighting or number of occurances, more is better
        if (currentOutputValue > valueToBeat) {
          valueChanged = true;
          bestOutputValue = outputClass;
          valueToBeat = currentOutputValue;
        }
      }

      if (!valueChanged) {
        throw new Exception("VALUE NOT CHANGED");
      }

      labels[0] = bestOutputValue;
    } else {
      // Else you are dealing with regression
      if (distanceWeighting) {
        continuousOutput /= continuousDenominator;
      } 

      labels[0] = continuousOutput;
    }
  }
}
