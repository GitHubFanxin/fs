package pers.xin.algorithm;

import pers.xin.core.attributeSelection.AttributeSelection;
import pers.xin.core.attributeSelection.FSAlgorithm;
import pers.xin.core.attributeSelection.MetaClassifier;
import pers.xin.core.entropy.ClassicalEntropy;
import pers.xin.core.entropy.Entropy;
import pers.xin.core.evaluation.FSEvaluation;
import pers.xin.utils.Tuple2;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.io.FileReader;
import java.util.*;

/**
 * ﻿
 *﻿ MOAYEDIKIA A, ONG K-L, BOO Y L，et al. Feature selection for high dimensional imbalanced class data using harmony
 * search[J]. Engineering Applications of Artificial Intelligence, 2017, 57: 38–49.
 * <p>Created by xin on 2018/8/22.
 */
public class SYMON extends FSAlgorithm {
    protected int HMS = 35;
    protected int NI = 200;
    protected double HMCR = 0.5;
    protected double PARmax = 0.9;
    protected double PARmin = 0.45;
    protected int rippleFactor = 2;
    protected int desiredSize;
    protected double desiredFactor;
    protected Classifier base = new J48();
    protected Random random = new Random();

    public SYMON(double desiredFactor) {
        this.desiredFactor = desiredFactor;
    }

    public SYMON(int HMS, int NI, double HMCR, double desiredFactor, Random random) {
        this.HMS = HMS;
        this.NI = NI;
        this.HMCR = HMCR;
        this.desiredFactor = desiredFactor;
        this.random = random;
    }

    public void setRippleFactor(int rippleFactor) {
        this.rippleFactor = rippleFactor;
    }

    public void setDesiredSize(int desiredSize) {
        this.desiredSize = desiredSize;
    }

    public void setBase(Classifier base) {
        this.base = base;
    }

    @Override
    protected List<Integer> select(Instances data) throws Exception {
        int length = data.numAttributes()-1;
        this.desiredSize = (int)(desiredFactor *length);
        double[] w = calculateSU(data);
        Tuple2<boolean[],Double>[] HM = new Tuple2[HMS];
        for (int i = 0; i < HMS; i++) {
            boolean[] boolSelected = randomGenerate(length);
            int[] selected = vectorToList(boolSelected);
            double value = measure(selected,data);
            HM[i] = new Tuple2<>(boolSelected,value);
        }
        for (int i = 0; i < NI; i++) {
            boolean[] NHV = new boolean[length];
            for (int j = 0; j < length; j++) {
                if (random.nextDouble() < HMCR) {
                    NHV[j] = HM[random.nextInt(HMS)]._1()[j];
                    if (random.nextDouble() < PAR(i)) {
                        NHV[j] = !NHV[j];
                    }
                }
                else {
                    NHV[j] = random.nextBoolean();
                }
            }
            boolean[] newNHV = vectorTune(w,NHV,rippleFactor,desiredSize);
            int[] selected = vectorToList(newNHV);
            double value = measure(selected,data);
            int worstIndex = findWorst(HM);
            if(HM[worstIndex]._2()<value){
                HM[worstIndex] = new Tuple2<>(newNHV,value);
            }
        }
        boolean[] bestNHV = HM[findBest(HM)]._1();
        List<Integer> reduction = vectorToArray(bestNHV);
        return reduction;
    }

    @Override
    public String getSetting() {
        return null;
    }

    @Override
    public void setOptions(String[] options) throws Exception {

    }

    private double PAR(int i){
        return PARmin + (PARmax-PARmin)*i/NI;
    }

    private boolean[] randomGenerate(int length){
        boolean[] l = new boolean[length];
        for (int i = 0; i < length; i++) {
            l[i] = random.nextBoolean();
        }
        return l;
    }

    private List<Integer> vectorToArray(boolean[] features) {
        ArrayList<Integer> array = new ArrayList<>();
        for (int i = 0; i < features.length; i++) {
            if (features[i]) array.add(i);
        }
        return array;
    }

    private int[] vectorToList(boolean[] features) {
        List<Integer> array = vectorToArray(features);
        int[] list = new int[array.size()];
        for (int i = 0; i < array.size(); i++) {
            list[i] = array.get(i);
        }
        return list;
    }

    private boolean[] arrayToVector(Collection<Integer> array, int length) {
        boolean[] vector = new boolean[length];
        for (Integer integer : array) {
            vector[integer] = true;
        }
        return vector;
    }

    private int findWorst(Tuple2<boolean[],Double>[] HM){
        double min = Double.POSITIVE_INFINITY;
        int minIndex = 0;
        for (int i = 0; i < HM.length; i++) {
            if(HM[i]._2()<min){
                min = HM[i]._2();
                minIndex = i;
            }
        }
        return minIndex;
    }

    private int findBest(Tuple2<boolean[],Double>[] HM){
        double max = Double.NEGATIVE_INFINITY;
        int maxIndex = 0;
        for (int i = 0; i < HM.length; i++) {
            if(HM[i]._2()>max){
                max = HM[i]._2();
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private double[] calculateSU(Instances data) {
        Entropy entropy = new ClassicalEntropy(data);
        int classIndex = data.classIndex();
        int numAttr = data.numAttributes()-1;
        double[] weights = new double[numAttr];
        double sum = 0;
        for (int i = 0; i < numAttr; i++) {
            weights[i] = entropy.SymmetricalUncertainty(i, classIndex);
            sum += weights[i];
        }
        for (int i = 0; i < weights.length; i++) {
            weights[i] /= sum;
        }
        return weights;
    }

    private boolean[] vectorTune(double[] w, boolean[] NHV, int r, int d) {
        PriorityQueue<Tuple2<Integer, Double>> Fs = new PriorityQueue<>((a, b) -> {
            if (a._2() == b._2()) return 0;
            return a._2() < b._2() ? 1 : -1;
        });
        PriorityQueue<Tuple2<Integer, Double>> Fu = new PriorityQueue<>(Comparator.comparingDouble(Tuple2::_2));
        for (int i = 0; i < NHV.length; i++) {
            if (NHV[i])
                Fs.add(new Tuple2<>(i, w[i]));
            else
                Fu.add(new Tuple2<>(i, w[i]));
        }
        if (Fs.size() == d) {
            rippleAdd(r, Fs, Fu);
            rippleRem(r, Fs, Fu);
        }
        while (Fs.size() > d) {
            rippleRem(r, Fs, Fu);
        }
        while (Fs.size() < d) {
            rippleAdd(r, Fs, Fu);
        }
        boolean[] newNHV = new boolean[NHV.length];
        for (Tuple2<Integer, Double> t : Fs) {
            newNHV[t._1()] = true;
        }
        return newNHV;
    }

    private void rippleAdd(int r, PriorityQueue<Tuple2<Integer, Double>> Fs,
                           PriorityQueue<Tuple2<Integer, Double>> Fu) {
        Set<Tuple2<Integer, Double>> rem = new HashSet<>(r);
        Set<Tuple2<Integer, Double>> add = new HashSet<>(r);
        for (int i = 0; i < r; i++) {
            add.add(Fu.poll());
        }
        add.remove(null);
        for (int i = 0; i < add.size()-1; i++) {
            rem.add(Fs.poll());
        }
        rem.remove(null);
        Fs.addAll(add);
        Fu.addAll(rem);
    }

    private void rippleRem(int r, PriorityQueue<Tuple2<Integer, Double>> Fs,
                           PriorityQueue<Tuple2<Integer, Double>> Fu) {
        Set<Tuple2<Integer, Double>> rem = new HashSet<>(r);
        Set<Tuple2<Integer, Double>> add = new HashSet<>(r);
        for (int i = 0; i < r; i++) {
            rem.add(Fs.poll());
        }
        rem.remove(null);
        for (int i = 0; i < rem.size()-1; i++) {
            add.add(Fu.poll());
        }
        add.remove(null);
        Fs.addAll(add);
        Fu.addAll(rem);
    }

    protected double measure(int[] selected, Instances data) throws Exception {
        Instances newData = instancesFilter(data, selected);
        FSEvaluation eval = new FSEvaluation(newData);
        Classifier copiedClassifier = AbstractClassifier.makeCopy(base);
        eval.crossValidateModel(copiedClassifier, newData, 5, new Random());
        return eval.areaUnderROC(1);
    }

    protected Instances instancesFilter(Instances data, int[] selected) throws Exception {
        int[] selectedAttr = new int[selected.length + 1];
        selectedAttr[selected.length] = data.classIndex();
        for (int i = 0; i < selected.length; i++) {
            selectedAttr[i] = selected[i];
        }
        Remove r = new Remove();
        r.setAttributeIndicesArray(selectedAttr);
        r.setInvertSelection(true);
        r.setInputFormat(data);
        return Filter.useFilter(data, r);
    }

    public static void main(String[] args) throws Exception {
        File file = new File("./dataset/ionosphere.arff");
        Instances data = new Instances(new FileReader((file)));
        data.setClassIndex(data.numAttributes()-1);
        SYMON symon = new SYMON(8);
        AttributeSelection as = new AttributeSelection(symon);
        MetaClassifier mc = new MetaClassifier();
        mc.setAttributeSelection(as);
        mc.setClassifier(new J48());
//        mc.buildClassifier(data);
        FSEvaluation eval = new FSEvaluation(data);
        eval.crossValidateModel(mc,data,5,new Random(1));
        System.out.println(eval.areaUnderROC(1));
    }
}
