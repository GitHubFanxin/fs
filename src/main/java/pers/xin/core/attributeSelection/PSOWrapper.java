package pers.xin.core.attributeSelection;

import pers.xin.algorithm.RSFSAIDXN;
import pers.xin.algorithm.TEST;
import pers.xin.core.evaluation.FSEvaluation;
import pers.xin.core.optimization.ISO;
import pers.xin.core.optimization.Position;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;

import java.io.File;
import java.io.FileReader;
import java.util.List;
import java.util.Random;

/**
 * Created by xin on 03/08/2018.
 */
public class PSOWrapper extends FSAlgorithm{
    protected FSAlgorithm fsAlgorithm;

    protected Classifier classifier;

    protected int maxIterate;

    protected int swarmSize;

    protected double trainRatio;

    protected Random random;

    protected Position bestP;

    public PSOWrapper(int maxIterate, int swarmSize, double trainRatio, Random random) {
        this.maxIterate = maxIterate;
        this.swarmSize = swarmSize;
        this.trainRatio = trainRatio;
        this.random = random;
    }

    public void setFsAlgorithm(FSAlgorithm fsAlgorithm){
        this.fsAlgorithm = fsAlgorithm;
    }

    public void setClassifier(Classifier classifier) {
        this.classifier = classifier;
    }

    @Override
    protected List<Integer> select(Instances data) throws Exception {
        FSOptimizeFunction function = new FSOptimizeFunction(data,trainRatio,random);
        function.setClassifier(classifier);
        function.setFsAlgorithm(fsAlgorithm);
        ISO iso = new ISO(swarmSize,maxIterate,0.6,2,2,random,function);
        bestP = iso.search();
        ((OptimizableFS)fsAlgorithm).setParams(bestP.get());
        return fsAlgorithm.select(data);
    }

    @Override
    public String getSetting() {
        return null;
    }

    @Override
    public void setOptions(String[] options) throws Exception {

    }

    public static void main(String[] args) throws Exception {
        File file = new File("./dataset/ionosphere.arff");
        Instances data = new Instances(new FileReader((file)));
        data.setClassIndex(data.numAttributes()-1);
        PSOWrapper test = new PSOWrapper(20,20,0.8,new Random());
        test.setClassifier(new J48());
        test.setFsAlgorithm(new RSFSAIDXN());
        AttributeSelection as = new AttributeSelection(test);
        MetaClassifier mc = new MetaClassifier();
        mc.setAttributeSelection(as);
        mc.setClassifier(new J48());
        FSEvaluation evaluation = new FSEvaluation(data);
        evaluation.crossValidateModel(mc,data,5,new Random(1));
        System.out.println(evaluation.pctCorrect());
    }
}
