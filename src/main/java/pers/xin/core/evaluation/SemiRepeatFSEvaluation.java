package pers.xin.core.evaluation;

import pers.xin.core.attributeSelection.AttributeSelection;
import pers.xin.core.attributeSelection.RepeatableMetaClassifier;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.core.Instances;

import java.util.Random;

/**
 * A help class, this evaluation can use a selected AttributeSelection. By reuse a AttributeSelection that already
 * built can reduce the evaluation time.
 * Created by xin on 10/06/2018.
 */
public class SemiRepeatFSEvaluation extends Evaluation {
    private double m_labeledRatio = 1;

    AttributeSelection[] m_attributeSelections;

    public SemiRepeatFSEvaluation(Instances data,double labeledRatio) throws Exception {
        super(data);
        this.m_labeledRatio = labeledRatio;
    }

    public SemiRepeatFSEvaluation(Instances data,double labeledRatio, CostMatrix costMatrix) throws Exception {
        super(data, costMatrix);
        this.m_labeledRatio = labeledRatio;
    }

    public void setAttributeSelections(AttributeSelection[] attributeSelections) {
        this.m_attributeSelections = attributeSelections;
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
            setPriors(train);
            Classifier copiedClassifier = AbstractClassifier.makeCopy(classifier);

            RepeatableMetaClassifier metaClassifier = new RepeatableMetaClassifier();
            metaClassifier.setAttributeSelection(m_attributeSelections[i]);
            metaClassifier.setClassifier(copiedClassifier);

            metaClassifier.buildClassifier(train);
            Instances test = data.testCV(numFolds, i);
            evaluateModel(metaClassifier, test, forPredictionsPrinting);
        }
        m_NumFolds = numFolds;

        if (classificationOutput != null) {
            classificationOutput.printFooter();
        }
    }

}
