package pers.xin.core.attributeSelection;

import pers.xin.core.attributeSelection.AttributeSelection;
import pers.xin.core.attributeSelection.FSAlgorithm;
import pers.xin.core.attributeSelection.MetaClassifier;
import pers.xin.core.attributeSelection.OptimizableFS;
import pers.xin.core.evaluation.FSEvaluation;
import pers.xin.core.optimization.Function;
import pers.xin.core.optimization.Position;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.util.Random;

/**
 * Created by xin on 2018/8/4.
 */
public class FSOptimizeFunction implements Function {
    protected FSAlgorithm fsAlgorithm;
    protected Classifier classifier;
    protected Instances data;
    protected double trainRatio;
    protected Random random;
    protected int dim;

    public FSOptimizeFunction(Instances data, double trainRatio, Random random) {
        this.data = data;
        this.trainRatio = trainRatio;
        this.random = random;
    }

    public void setFsAlgorithm(FSAlgorithm fsAlgorithm) throws Exception {
        if (fsAlgorithm instanceof OptimizableFS){
            this.fsAlgorithm = fsAlgorithm;
            dim = ((OptimizableFS)fsAlgorithm).numParams();
        }
        else throw new Exception("algorithm can not be optimized!");
    }

    public void setClassifier(Classifier classifier) {
        this.classifier = classifier;
    }

    @Override
    public int dimension() {
        return dim;
    }

    @Override
    public double computeFitness(Position params) {
        double fitness = 0;
        try {
            fitness = measure(params.get());
        } catch (Exception e) {
            e.printStackTrace();
        }
        return fitness;
    }

    private double measure(double[] params) throws Exception {
        data.randomize(random);
        int numtrain = (int) (data.numInstances()*trainRatio);
        Instances train = new Instances(data,numtrain);
        Instances test = new Instances(data,data.numInstances()-numtrain);
        for (int i = 0; i < data.numInstances(); i++) {
            if(i<numtrain) train.add(data.get(i));
            else test.add(data.get(i));
        }
        FSAlgorithm algorithm = FSAlgorithm.makeCopy(fsAlgorithm);
        ((OptimizableFS)algorithm).setParams(params);
        Classifier copiedClassifier = AbstractClassifier.makeCopy(classifier);
        AttributeSelection as = new AttributeSelection(algorithm);
        MetaClassifier mc = new MetaClassifier();
        mc.setClassifier(copiedClassifier);
        mc.setAttributeSelection(as);
        mc.buildClassifier(train);
        FSEvaluation evaluation = new FSEvaluation(data);
        evaluation.evaluateModel(mc,test);
        return evaluation.areaUnderPRC(1);
    }
}
