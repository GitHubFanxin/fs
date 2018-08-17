package pers.xin.core.attributeSelection;

import pers.xin.core.evaluation.FSEvaluation;
import pers.xin.core.optimization.Function;
import pers.xin.core.optimization.Position;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.io.Serializable;
import java.util.Random;

/**
 * Created by xin on 2018/8/4.
 */
public abstract class AbstractFSFunction implements Function,Serializable {
    protected FSAlgorithm fsAlgorithm;
    protected Classifier classifier;
    protected Instances data;
    protected double trainRatio;
    protected Random random;
    protected int dim;

//    public AbstractFSFunction(Instances data, double trainRatio, Random random) {
//        this.data = data;
//        this.trainRatio = trainRatio;
//        this.random = random;
//    }


    public void setData(Instances data) {
        this.data = data;
    }

    public void setTrainRatio(double trainRatio) {
        this.trainRatio = trainRatio;
    }

    public void setRandom(Random random) {
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
        double fitness = Double.NEGATIVE_INFINITY;
        try {
            fitness = measure(params.get());
        } catch (Exception e) {
            System.out.println(e.getMessage());
            fitness = Double.NEGATIVE_INFINITY;
        }
        return fitness;
    }

    protected abstract double measure(double[] params) throws Exception;
}
