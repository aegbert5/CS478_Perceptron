import java.util.*;

public class FeatureDivisor {

  int featureColumn;
  String name;
  List<Integer> remainingDataIndexes; // Indexes of features who satisfy the conditions of the feature divisor
  List<Integer> remainingFeatures; // Set of feature columns left to check
  Map<Integer, FeatureDivisor> divisorMap;
  Integer outputValue; // Only used for leaf nodes, or when features run out
  boolean isLeafNode; // Leaf Node until proven otherwise

  public FeatureDivisor(String name) {
    this.featureColumn = -1;
    this.name = name;
    this.remainingDataIndexes = new LinkedList();
    this.remainingFeatures = new LinkedList();
    this.divisorMap = new HashMap();
    this.outputValue = 0;
    this.isLeafNode = true;
  }

  @Override
  public int hashCode() {
    final int prime = 13;
    int result = 1;
    result = prime * result + ((name == null) ? 0 : name.hashCode());
    result = prime * result + ((remainingDataIndexes == null) ? 0 : remainingDataIndexes.hashCode());
    result = prime * result + ((remainingFeatures == null) ? 0 : remainingFeatures.hashCode());
    result = prime * result + ((divisorMap == null) ? 0 : divisorMap.hashCode());
    result = prime * result + ((outputValue == null) ? 0 : outputValue.hashCode());
    return result;
  }
}
