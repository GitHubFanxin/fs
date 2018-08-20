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
import weka.classifiers.trees.J48;
import weka.core.Instances;

import java.io.Serializable;
import java.util.Random;

/**
 * Created by xin on 2018/8/4.
 */
public class PctFSFunction extends AbstractFSFunction implements Serializable{


    private static final long serialVersionUID = 8366964569106450942L;

    @Override
    protected double measure(double[] params) throws Exception {
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
        Classifier copiedClassifier1 = AbstractClassifier.makeCopy(classifier);

        AttributeSelection as = new AttributeSelection(algorithm);
        MetaClassifier mc = new MetaClassifier();
        mc.setClassifier(copiedClassifier);
        mc.setAttributeSelection(as);
        mc.buildClassifier(train);
        FSEvaluation evaluation = new FSEvaluation(data);
        evaluation.evaluateModel(mc,test);

        copiedClassifier1.buildClassifier(train);
        FSEvaluation evaluation1 = new FSEvaluation(data);
        evaluation1.evaluateModel(copiedClassifier1,test);
//        double AUC = evaluation.areaUnderROC(1);
//        double AUC2= evaluation1.areaUnderROC(1);
        double pct = evaluation.pctCorrect();
        double pct2 = evaluation1.pctCorrect();
        return pct-pct2;
    }
}
