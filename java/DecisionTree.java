import javafx.util.Pair;
import java.util.Random;
import java.util.*;

public class DecisionTree extends SupervisedLearner {

  Random rand;
  FeatureDivisor rootDivisor;
  boolean printTree = false;
  boolean debug = false;
  Map<Integer, Integer> averageValueForFeature;

  public DecisionTree(Random rand) {
    this.rand = rand;
  }      

  public void train(Matrix features, Matrix labels) throws Exception {
    this.rootDivisor = new FeatureDivisor("Root");
    this.averageValueForFeature = new HashMap();

    int numOutputClasses = labels.valueCount(0);

    int numFeatures = features.row(0).length;
    int numInstances = features.rows();

    // Check to make sure that the data is all nominal inputs and outputs
    for (int i = 0; i < numFeatures; ++i) {
      if (features.valueCount(i) < 2) {
        throw new Exception("Decision Tree Cannot Accept Features with continuous values");
      }
    }

    if (numOutputClasses < 2) {
      throw new Exception("Decision Tree Cannot Accept Data that has continuous outputs");
    }

    // All of the indexes are valid at first
    List<Integer> validIndexes = new LinkedList();
    for (int i = 0; i < numInstances; ++i) {
      validIndexes.add(i);
    }

    // All the features are remaining
    List<Integer> remainingFeatures = new LinkedList();
    for (int i = 0; i < numFeatures; ++i) {
      remainingFeatures.add(i);
      double averageValue = 0.0;
      for (int j = 0; j < numInstances; ++j) {
        averageValue += features.get(j, i);
      }

      averageValue /= numInstances;
      averageValueForFeature.put(i, (int) averageValue);
    }

    rootDivisor.remainingDataIndexes = validIndexes;
    rootDivisor.remainingFeatures = remainingFeatures;

    generateDecisionTree(numOutputClasses, numFeatures, features, labels, rootDivisor);

    if (printTree) {
      printTree(rootDivisor, 1, features, labels);
      System.out.println();
    }

    int[] nodeCount = { 0 };
    int[] treeDepth = { 0 };

    countNodesAndDepth(rootDivisor, nodeCount, treeDepth, 0);
    System.out.println("BEFORE: " + nodeCount[0] + ", " + treeDepth[0]);

    //reduceErrorPrune(rootDivisor, features, labels);

    nodeCount[0] = 0;
    treeDepth[0] = 0;

    countNodesAndDepth(rootDivisor, nodeCount, treeDepth, 0);
    System.out.println("AFTER: " + nodeCount[0] + ", " + treeDepth[0]);
  }

  // Prints the tree's decision path
  private void printTree(FeatureDivisor currentDivisor, int level, Matrix features, Matrix labels) {
    if (!currentDivisor.isLeafNode) {
      System.out.print(" -> " + currentDivisor.name + ": ");
    }

    System.out.print(" => " + labels.attrValue(0, currentDivisor.outputValue));

    if (level < 3) {
    for (Integer featureValue : currentDivisor.divisorMap.keySet()) {
      System.out.println();
      for (int i = 0; i < level; ++i) {
        System.out.print("  ");
      }
      System.out.print(features.attrValue(currentDivisor.featureColumn, featureValue));
      printTree(currentDivisor.divisorMap.get(featureValue), level+1, features, labels);
    }
    }
  }

  private void setMostPrevalentOutput(FeatureDivisor currentDivisor, List<Integer> remainingIndexes, Matrix features, Matrix labels) {
    Map<Integer, Integer> outputFrequency = new HashMap(); 

    int numInstances = remainingIndexes.size();

    for (int j = 0; j < numInstances; ++j) {
      int instanceIndex = remainingIndexes.get(j);
      double[] currentInstance = features.row(instanceIndex);
      int currentOutput = (int) labels.get(instanceIndex, 0);
  
      int currentTotal = outputFrequency.getOrDefault(currentOutput, 0);
      outputFrequency.put(currentOutput, ++currentTotal);
    }

    int mostFrequentOutput = -1;
    int mostFrequencyOutputCount = 0;

    for (Integer i : outputFrequency.keySet()) {
      if (outputFrequency.get(i) > mostFrequencyOutputCount) {
        mostFrequencyOutputCount = outputFrequency.get(i);
        mostFrequentOutput = i;
      }
    }

    currentDivisor.outputValue = mostFrequentOutput;
  }

  // Returns whether the node is a leaf node because there is only one outputValue
  // Otherwise, this will populate the featureCount and featureOutputCount maps that tally how many times a feature, feature/output combo appears
  private boolean isLeafNode(FeatureDivisor currentDivisor, List<Integer> currentIndexes, int numInstances, Matrix features, Matrix labels, Map<Pair<Integer, Integer>, Integer> featureCount, Map<Triple<Integer, Integer, Integer>, Integer> featureOutputCount, List<Integer> remainingFeatures) {

    // Is a leaf node unless proven otherwise
    boolean isLeafNode = true; 
    Integer leafNodeOutput = -1;

    int numRemainingFeatures = remainingFeatures.size();

    // Count the amount each feature comes up as well as the feature/output count combination
    for (int j = 0; j < numInstances; ++j) {
      int instanceIndex = currentIndexes.get(j);
      double[] currentInstance = features.row(instanceIndex);
      int currentOutput = (int) labels.get(instanceIndex, 0);

      // Determines whether this is a leaf node
      if (leafNodeOutput == -1) {
        leafNodeOutput = currentOutput;
      } else if (currentOutput != leafNodeOutput) {
        isLeafNode = false;
      }
       
      for (int i = 0; i < numRemainingFeatures; ++i) { 
        int remainingFeatureIndex = remainingFeatures.get(i);
        double value = currentInstance[remainingFeatureIndex];

        int currentFeatureValue = (int) value;
        if (value == Double.MAX_VALUE) {
          currentFeatureValue = averageValueForFeature.get(remainingFeatureIndex);
        }
       
        Pair<Integer, Integer> currentFeaturePair = new Pair(remainingFeatureIndex, currentFeatureValue);
        int currentFeatureCount = featureCount.getOrDefault(currentFeaturePair, 0);
        featureCount.put(currentFeaturePair, ++currentFeatureCount);

        Triple<Integer, Integer, Integer> featureOutputKey = new Triple(remainingFeatureIndex, currentFeatureValue, currentOutput);
        int pairCount = featureOutputCount.getOrDefault(featureOutputKey, 0);
        featureOutputCount.put(featureOutputKey, ++pairCount);
      }
    }
    
    // If this is a leaf node, there is not more splitting to do
    currentDivisor.isLeafNode = isLeafNode;
    return isLeafNode;
  }

  // Recursively generates the decision tree based off what is dividing the dataset currently
  private void generateDecisionTree(int numOutputClasses, int numFeatures, Matrix features, Matrix labels, FeatureDivisor currentDivisor) {

    List<Integer> currentIndexes = currentDivisor.remainingDataIndexes;
    int numInstances = currentIndexes.size();

    List<Integer> remainingFeatures = currentDivisor.remainingFeatures;
    int numRemainingFeatures = remainingFeatures.size();
    
    // Set this node's output value as the most prevalent of the outputs
    setMostPrevalentOutput(currentDivisor, currentIndexes, features, labels);

    // If there are no more features to visit, then there is no more splitting to do
    if (numRemainingFeatures == 0) {
      return;
    }
  
    // Counts the instances where the feature values are the same
    Map<Pair<Integer, Integer>, Integer> featureCount = new HashMap();

    // Counts the instances where the feature values AND output are the same
    Map<Triple<Integer, Integer, Integer>, Integer> featureOutputCount = new HashMap();

    if (isLeafNode(currentDivisor, currentIndexes, numInstances, features, labels, featureCount, featureOutputCount, remainingFeatures)) {
      return;
    }

//    for (Pair t : featureCount.keySet()) {
//      System.out.println(t + " -> " + featureCount.get(t));
//    }
//
//    for (Triple t : featureOutputCount.keySet()) {
//      System.out.println(t + " -> " + featureOutputCount.get(t));
//    }

    // Get the feature that returns the lowest info at that level
    int bestFeatureInfo = -1;
    double bestFeatureInfoValue = Integer.MAX_VALUE;

    for (int i = 0; i < numRemainingFeatures; ++i) {
      int remainingFeatureIndex = remainingFeatures.get(i);
      int numFeatureValues = features.valueCount(remainingFeatureIndex);
      
      double featureInfo = 0;

      if (debug) {
        String currentName = features.attrName(remainingFeatureIndex);
        System.out.print("INFO(" + currentName + "): ");
      }

      for (int j = 0; j < numFeatureValues; ++j) {
        Pair<Integer, Integer> currentFeaturePair = new Pair(remainingFeatureIndex, j);
        int numInstancesWithValue = featureCount.getOrDefault(currentFeaturePair, 0);
        double multConstant = (double) numInstancesWithValue / (double) numInstances;
      
        if (debug) {
          System.out.print(numInstancesWithValue + "/" + numInstances + "*(");
        }

        if (numInstancesWithValue != 0) {
          for (int a = 0; a < numOutputClasses; ++a) {
            Triple<Integer, Integer, Integer> currentOutputTriple = new Triple(remainingFeatureIndex, j, a);
            int numInstancesWithValueAndOutput = featureOutputCount.getOrDefault(currentOutputTriple, 0);
  
            double ratio = (double) numInstancesWithValueAndOutput / (double) numInstancesWithValue;
            if (numInstancesWithValueAndOutput != 0) {
              featureInfo += multConstant * (-ratio) * (Math.log(ratio) / Math.log(2));
            }
 
            if (debug) {
              String ratioString = numInstancesWithValueAndOutput + "/" + numInstancesWithValue;
              System.out.print("-" + ratioString + " * log(" + ratioString + ")");
            }
          }
          if (debug) {
          System.out.println(")");
  
            if (j != numFeatureValues - 1) {
              System.out.print(" + ");
            } else {
              System.out.println();
            }
          }
        }
      }

      if (debug) {
        System.out.println(features.attrName(remainingFeatureIndex) + " -> " + featureInfo);
      }

      if (featureInfo < bestFeatureInfoValue) {
        bestFeatureInfoValue = featureInfo;
        bestFeatureInfo = remainingFeatureIndex;
      }
    }

    String featureName = features.attrName(bestFeatureInfo);
    currentDivisor.featureColumn = bestFeatureInfo;
    currentDivisor.name = featureName;

    // Split up the current instances of data by their feature values 
    //int numFeatureValues = features.valueCount(bestFeatureInfo);
    for (int i = 0; i < numInstances; ++i) {
      int instanceIndex = currentIndexes.get(i);
      double[] currentInstance = features.row(instanceIndex);

      int currentOutput = (int) labels.get(instanceIndex, 0); 
      int currentFeatureValue = (int) currentInstance[bestFeatureInfo];
      String currentFeatureName = features.attrValue(bestFeatureInfo, currentFeatureValue);

      FeatureDivisor selectedDivisor = currentDivisor.divisorMap.getOrDefault(currentFeatureValue, new FeatureDivisor("LEAF"));
      selectedDivisor.remainingDataIndexes.add(instanceIndex);
      currentDivisor.divisorMap.put(currentFeatureValue, selectedDivisor);
    }

    if (debug) {
      System.out.println("DIVIDED: " + currentDivisor.name);
    }

    // For each divisor value for the feature, update the tree below it
    int numFeatureValues = features.valueCount(bestFeatureInfo);
    for (int i = 0; i < numFeatureValues; ++i) {
      FeatureDivisor nextDivisor = currentDivisor.divisorMap.getOrDefault(i, new FeatureDivisor("leaf node")); // If a feature value does not occur in the instances, then give it the most likely value from the parent set
      if (debug) {
        System.out.println("CHECKING... " + featureName + "-" + features.attrValue(bestFeatureInfo, i));
      }
      nextDivisor.featureColumn = i;
      for (int j = 0; j < numRemainingFeatures; ++j) {
        // Add everything but the current feature
        if (bestFeatureInfo != j) {
          nextDivisor.remainingFeatures.add(j);
        }
      }
      generateDecisionTree(numOutputClasses, numFeatures, features, labels, nextDivisor);
    }
  }

  private void countNodesAndDepth(FeatureDivisor currentDivisor, int[] nodeCount, int[] treeDepth, int currentDepth) {
    if (currentDivisor != null) {
      nodeCount[0]++;
      boolean isLeafNode = true; // Some of the objects dont have any sub-trees
      for (Integer value : currentDivisor.divisorMap.keySet()) {
        isLeafNode = false;
        FeatureDivisor newDivisor = currentDivisor.divisorMap.get(value);
        countNodesAndDepth(newDivisor, nodeCount, treeDepth, currentDepth + 1);
      }

      if (isLeafNode) {
        if (currentDepth > treeDepth[0]) {
          treeDepth[0] = currentDepth;
        }
      }
    } else {
      if (currentDepth > treeDepth[0]) {
        treeDepth[0] = currentDepth;
      }
    }
  }

  private void reduceErrorPrune(FeatureDivisor currentDivisor, Matrix features, Matrix labels) throws Exception {

    if (currentDivisor != null) {
      if (!currentDivisor.isLeafNode) {
        for (Integer value : currentDivisor.divisorMap.keySet()) {
          FeatureDivisor prunedDivisor = currentDivisor.divisorMap.get(value);
          double previousAccuracy = measureAccuracy(features, labels, null);
  
          currentDivisor.divisorMap.put(value, null);
          double prunedAccuracy = measureAccuracy(features, labels, null);
  
          // Only keep the prune when the accuracy doesn't change, search in sub-tree for more prunes
          if (prunedAccuracy <= previousAccuracy) {
            currentDivisor.divisorMap.put(value, prunedDivisor);
            reduceErrorPrune(prunedDivisor, features, labels);
          }
        }
      }
    }
  }

  public void predict(double[] features, double[] labels) throws Exception {

    FeatureDivisor currentDivisor = rootDivisor;
    
    while (!currentDivisor.isLeafNode) {
      int columnToCheck = currentDivisor.featureColumn;
      double value = (int) features[columnToCheck];

      int featureValue = (int) value;
      if (value == Double.MAX_VALUE) {
        featureValue = averageValueForFeature.get(columnToCheck);
      }

      FeatureDivisor nextDivisor = currentDivisor.divisorMap.get(featureValue);
      if (nextDivisor != null) {
        currentDivisor = nextDivisor;
      } else {
        break;
      }
    }

    labels[0] = currentDivisor.outputValue;
  }

}
