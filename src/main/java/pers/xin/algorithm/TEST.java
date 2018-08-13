package pers.xin.algorithm;

import it.unimi.dsi.fastutil.ints.IntIterator;
import it.unimi.dsi.fastutil.ints.IntOpenHashSet;
import it.unimi.dsi.fastutil.ints.IntSet;
import it.unimi.dsi.fastutil.objects.Object2IntMap;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import pers.xin.core.attributeSelection.AttributeSelection;
import pers.xin.core.attributeSelection.FSAlgorithm;
import pers.xin.core.attributeSelection.MetaClassifier;
import pers.xin.core.attributeSelection.OptimizableFS;
import pers.xin.core.entropy.IINM;
import pers.xin.core.evaluation.FSEvaluation;
import pers.xin.core.neighbor.Neighborhood;
import pers.xin.core.neighbor.NeighborhoodInfinity;
import weka.classifiers.trees.J48;
import weka.core.*;

import java.io.File;
import java.io.FileReader;
import java.util.*;
import java.util.stream.Collectors;

/**
 * 矩阵统计频数
 * Created by xin on 06/03/2018.
 */
public class TEST extends FSAlgorithm implements OptimizableFS {
    private int count=10;

    private double delta = 0.125;

    public double getDelta() {
        return delta;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }

    @Override
    protected List<Integer> select(Instances data) throws Exception {

        ArrayList<Integer> reduct = new ArrayList<>();

//        NeighborhoodEntropy ne = new NeighborhoodEntropy(delta,labeledData,data);

        IINM ne = new IINM(delta,data,data);

        int numAttr = data.numAttributes()-1;

        Set<Integer> allIndices = new HashSet<>();
        for (int i = 0; i < numAttr; i++) {
            allIndices.add(i);
        }
        Set<Integer> restIndices = new HashSet<>();
        restIndices.addAll(allIndices);

        int numdata = data.numInstances();
        /**
         * 邻域集合
         */
        boolean[][] neighborSets = new boolean[numdata][numdata];
        int[] generalDecision = new int[numdata];
        Neighborhood neighbor = new NeighborhoodInfinity(data,delta);

        for (int i = 0; i < numdata; i++) {
            int g = 0;
            Instance first = data.get(i);
            for (int j = i; j < numdata; j++) {
                Instance second = data.get(j);
                if(neighbor.isNeighbor(first,second)){
                    neighborSets[i][j]=true;
                    neighborSets[j][i]=true;
                    generalDecision[i]=generalDecision[i]|(1<<(int)second.classValue());
                    generalDecision[j]=generalDecision[j]|(1<<(int)first.classValue());
                }
            }
            generalDecision[i]=g;
        }


        double maxVal = Double.NEGATIVE_INFINITY;
        int bestAttrIndex = -1;
        for (Integer index : allIndices) {
            double relevance = ne.mutualInformation(data.classIndex(),index);
            if(relevance>maxVal){
                bestAttrIndex = index;
                maxVal = relevance;
            }
        }
        reduct.add(bestAttrIndex);
        restIndices.remove(bestAttrIndex);

        /** 计算可区分属性(删除最佳属性后) */
        Object2IntMap<IntSet> discernibilityMatrix = new Object2IntOpenHashMap<>();
        int[] redundancy = new int[numAttr];
        int element = 0;
        int[] attributeFrequency = new int[numAttr];
        for (int i = 0; i < numdata; i++) {
            Instance first = data.get(i);
            for (int j = 0; j < i; j++) {
                Instance second = data.get(j);
//                if(generalDecision[i] != generalDecision[j] && (!neighborSets[i][j])){
//                if(generalDecision[i]!=0 && generalDecision[j]!=0 && generalDecision[i] != generalDecision[j] &&
//                        (!neighborSets[i][j])){
                if(data.get(i).classValue() != data.get(j).classValue()&& (!neighborSets[i][j])){
//                if(data.get(i).classValue() != data.get(j).classValue()){
                    /**检验最佳属性**/
                    if(!neighbor.isNeighborOnAttr(bestAttrIndex,first,second)){
                        for (int k = 0; k < data.numAttributes() - 1; k++) {
                            if(!neighbor.isNeighborOnAttr(k,first,second)){
                                redundancy[k]++;
                            }
                        }
                        continue;
                    }
                    IntSet discernibilityAttr = new IntOpenHashSet();
                    for (int k = 0; k < data.numAttributes() - 1; k++) {
                        if(!neighbor.isNeighborOnAttr(k,first,second)){
                            discernibilityAttr.add(k);
                        }
                    }
                    if(!discernibilityAttr.isEmpty()){
                        int count = discernibilityMatrix.getOrDefault(discernibilityAttr,0);
                        discernibilityMatrix.put(discernibilityAttr,count+1);
                        element++;
                    }
                }
            }
        }
        discernibilityMatrix.remove(new HashSet<HashSet<Integer>>());

        for (Object2IntMap.Entry<IntSet> entry : discernibilityMatrix.object2IntEntrySet()) {
            for(IntIterator i = entry.getKey().iterator(); i.hasNext();){
                attributeFrequency[i.nextInt()] += entry.getIntValue();
            }
        }



        while (discernibilityMatrix.size()>0){
//        while (reduct.size()<count){
            bestAttrIndex = -1;
            maxVal = Double.NEGATIVE_INFINITY;



            for (int i : restIndices) {
//                double relevance = ne.mutualInformation(data.classIndex(),i);
                double sig = attributeFrequency[i]*1.0/redundancy[i];
                if(sig>maxVal){
                    bestAttrIndex = i;
                    maxVal = sig;
                }
            }
            if(bestAttrIndex != -1){

//                System.out.println(bestAttrIndex +"\t"+attributeFrequency[bestAttrIndex]);
                reduct.add(bestAttrIndex);
                restIndices.remove(bestAttrIndex);
                Object2IntMap<IntSet> tmp = new Object2IntOpenHashMap<>();
                for (Object2IntMap.Entry<IntSet> entry : discernibilityMatrix.object2IntEntrySet()) {
                    if(!entry.getKey().contains(bestAttrIndex)){
                        for(IntIterator i = entry.getKey().iterator(); i.hasNext();){
                            redundancy[i.nextInt()] += entry.getIntValue();
                        }
                        tmp.put(entry.getKey(),entry.getIntValue());
                    }
                }
                discernibilityMatrix = tmp;
                element = 0;
                attributeFrequency = new int[numAttr];
                for (Object2IntMap.Entry<IntSet> entry : discernibilityMatrix.object2IntEntrySet()) {
                    element += entry.getIntValue();
                    for(IntIterator i = entry.getKey().iterator(); i.hasNext();){
                        attributeFrequency[i.nextInt()] += entry.getIntValue();
                    }
                }

                restIndices.clear();
                for (IntSet indices : discernibilityMatrix.keySet()) {
                    restIndices.addAll(indices);
                }
            }else break;
        }
//        System.out.println(reduct.stream().map(x->""+x).collect(Collectors.joining(",")));
        return reduct;
    }

    @Override
    public String getSetting() {
        return "D"+delta;
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        System.gc();
    }

    double sigmoid(double x){
        return 1.0/(1+Math.exp(-x));
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String deltaString = Utils.getOption('D',options);
        if(deltaString.length()!=0){
            setDelta(Double.parseDouble(deltaString));
        }else {
            setDelta(0.125);
        }
    }

    @Override
    public String[] getOptions() {
        Vector<String> options = new Vector<>();
        options.add("-D");
        options.add(""+delta);
        return options.toArray(new String[0]);
    }


    @Override
    public int numParams() {
        return 1;
    }

    @Override
    public void setParams(double... params) {
        this.delta = params[0];
    }

    public static void main(String[] args) throws Exception {
        File file = new File("./dataset/ionosphere.arff");
        Instances data = new Instances(new FileReader((file)));
        data.setClassIndex(data.numAttributes()-1);
        TEST test = new TEST();
        test.setOptions(Utils.splitOptions("-D 0.25"));
        AttributeSelection as = new AttributeSelection(test);
        MetaClassifier mc = new MetaClassifier();
        mc.setAttributeSelection(as);
        mc.setClassifier(new J48());
        FSEvaluation evaluation = new FSEvaluation(data);
        evaluation.crossValidateModel(mc,data,5,new Random(1));
        System.out.println(evaluation.pctCorrect());
    }
}
