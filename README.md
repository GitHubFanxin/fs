Feature selection Experiment APIs based on WEKA
======
* 继承FSAlgorithm类实现特征选择算法Algorithm；

   **List<Integer> select(Instances data)方法**：特征选择方法，需返回选择属性索引数组<br/>
   **void setOptions(String[] options)**：参数设置方法，参考TEST算法实现<br />
   **String[] getOptions()**：获得参数方法，参考TEST算法实现<br/>
* 通过RepeatableExperiment类实例进行Filter特征选择算法实验；

```java
RepeatableExperiment e = new RepeatableExperiment(10,1);
e.setDataFilePath("./dataset");
Algorithm algorithm = new Algorithm();
Algorithm.setOptions(Utils.splitOptions("****"));
Output.setFolder("./output/"+ *** +"/");
e.setFSAlgorithm(algorithm);
J48 j48 = new J48();
IBk iBk = new IBk();
iBk.setKNN(3);
Classifier[] cs = new Classifier[]{j48, iBk};
e.setClassifiers(cs);
e.run(RepeatableExperiment.DATA_CLASSIFIERS);
```
