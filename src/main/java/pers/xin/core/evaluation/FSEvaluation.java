package pers.xin.core.evaluation;

import pers.xin.core.attributeSelection.AttributeSelection;
import pers.xin.core.attributeSelection.FSMeasure;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.core.Instances;
import weka.core.WeightedInstancesHandler;

import java.util.Random;

/**
 * Created by xin on 10/06/2018.
 */
public class FSEvaluation extends Evaluation {
    String[] m_crossValidateReductions;

    String m_reduction;

    double[] m_crossValidateTime;

    double m_time;


    public FSEvaluation(Instances data) throws Exception {
        super(data);
    }

    public FSEvaluation(Instances data, CostMatrix costMatrix) throws Exception {
        super(data, costMatrix);
    }

    public String[] getCrossValidateReductions() {
        return m_crossValidateReductions;
    }

    public String getReduction() {
        return m_reduction;
    }

    public double[] getCrossValidateTime() {
        return m_crossValidateTime;
    }

    public double getTime() {
        return m_time;
    }

    /**
     * Performs a (stratified if class is nominal) cross-validation for a
     * classifier on a set of instances. Now performs a deep copy of the
     * classifier before each call to buildClassifier() (just in case the
     * classifier is not initialized properly).
     *
     * @param classifier the classifier with any options set.
     * @param data the data on which the cross-validation is to be performed
     * @param numFolds the number of folds for the cross-validation
     * @param random random number generator for randomization
     * @param forPredictionsPrinting varargs parameter that, if supplied, is
     *          expected to hold a
     *          weka.classifiers.evaluation.output.prediction.AbstractOutput
     *          object
     * @throws Exception if a classifier could not be generated successfully or
     *           the class is not defined
     */
    public void crossValidateModel(Classifier classifier, Instances data,
                                   int numFolds, Random random, Object... forPredictionsPrinting)
            throws Exception {

        m_crossValidateTime = new double[numFolds];
        m_crossValidateReductions = new String[numFolds];

        // Make a copy of the data we can reorder
        data = new Instances(data);
        data.randomize(random);
        if (data.classAttribute().isNominal()) {
            data.stratify(numFolds);
        }

        // We assume that the first element is a
        // weka.classifiers.evaluation.output.prediction.AbstractOutput object
        AbstractOutput classificationOutput = null;
        if (forPredictionsPrinting.length > 0) {
            // print the header first
            classificationOutput = (AbstractOutput) forPredictionsPrinting[0];
            classificationOutput.setHeader(data);
            classificationOutput.printHeader();
        }

        // Do the folds
        for (int i = 0; i < numFolds; i++) {
            Instances train = data.trainCV(numFolds, i, random);
            setPriors(train);
            Classifier copiedClassifier = AbstractClassifier.makeCopy(classifier);
            copiedClassifier.buildClassifier(train);
            Instances test = data.testCV(numFolds, i);
            evaluateModel(copiedClassifier, test, forPredictionsPrinting);
            m_crossValidateTime[i] = m_time;
            m_crossValidateReductions[i] = m_reduction;
        }
        m_NumFolds = numFolds;

        if (classificationOutput != null) {
            classificationOutput.printFooter();
        }
    }

    @Override
    public double[] evaluateModel(Classifier classifier, Instances data, Object... forPredictionsPrinting) throws Exception {
        double[] result = super.evaluateModel(classifier, data, forPredictionsPrinting);
        if(classifier instanceof FSMeasure){
            m_time = ((FSMeasure) classifier).measureSelectionTime();
            m_reduction = ((FSMeasure) classifier).getReductionString();
        }
        return result;
    }
}
