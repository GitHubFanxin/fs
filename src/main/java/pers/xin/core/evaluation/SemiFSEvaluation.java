package pers.xin.core.evaluation;

import pers.xin.core.attributeSelection.FSMeasure;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Random;

/**
 * Evaluate FSAlgorithm only one time. First, split data set into labeled part and unlabeled part according to the gaven
 * label ratio. Then automatically generate train sets and test sets for folds cross validate from the labeled part.
 * Created by xin on 10/06/2018.
 */
public class SemiFSEvaluation extends Evaluation {
    private double m_labeledRatio = 1;

    String[] m_crossValidateReductions;

    String m_reduction;

    double[] m_crossValidateTime;

    double m_time;


    public SemiFSEvaluation(Instances data,double labeledRatio) throws Exception {
        super(data);
        this.m_labeledRatio = labeledRatio;
    }

    public SemiFSEvaluation(Instances data,double labeledRatio, CostMatrix costMatrix) throws Exception {
        super(data, costMatrix);
        this.m_labeledRatio = labeledRatio;
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
     * The data is divided into 2 equal parts. One part is labeled data, another is unlabeled data. Labeled part is
     * divided into numFolds parts, each part used to test algorithm and other parts and the unlabeled part make up
     * training set.
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

        int numInstances = data.numInstances();
        Instances l_data = data.stringFreeStructure();
        Instances u_data = data.stringFreeStructure();
        for (int i = 0; i < numInstances; i++) {
            if(i<numInstances*m_labeledRatio) l_data.add(data.get(i));
            else u_data.add(data.get(i));
        }
        for (int i = 0; i < u_data.size(); i++) {
            u_data.get(i).setClassMissing();
        }

        // Make a copy of the data we can reorder
        data = new Instances(l_data);
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
            train.addAll(u_data);

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

    public double[] timeMeasures(){
        return m_crossValidateTime;
    }

    public double timeMeasure(){
        return Arrays.stream(m_crossValidateTime).reduce(Double::sum).orElse(0)/m_crossValidateTime.length;
    }
}
