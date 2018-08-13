package pers.xin.core.evaluation;

import pers.xin.core.attributeSelection.AttributeSelection;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.core.Instances;
import weka.core.WeightedInstancesHandler;

import java.util.Arrays;
import java.util.Random;

/**
 * Filter Feature Selection use this Evaluation. This evaluation can reuse the result of attribute selection on.First,
 * split data set into labeled part and unlabeled part according to the gaven
 * label ratio. Then automatically generate train sets and test sets for folds cross validate from the labeled part.
 * different base classifier.
 * Created by xin on 11/06/2018.
 */
public class SemiMultipleFSEvaluation extends MultipleFSEvaluation {
    private double m_labeledRatio = 1;

    public SemiMultipleFSEvaluation(Instances data, double labeledRatio) {
        super(data);
        this.m_labeledRatio = labeledRatio;
    }

    public AttributeSelection[] generateCrossFS(AttributeSelection attributeSelection, Instances data,
                                                int numFolds, Random random, Object... forPredictionsPrinting)
            throws Exception {

        m_attributeSelections = new AttributeSelection[numFolds];
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
            if (i < numInstances * m_labeledRatio) l_data.add(data.get(i));
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

            AttributeSelection copiedAttributeSelection = AttributeSelection.makeCopy(attributeSelection);
            m_attributeSelections[i] = buildAttributeSelection(copiedAttributeSelection, train);
            m_reduction = m_attributeSelections[i].toResultsString();
            m_crossValidateTime[i] = m_time;
            m_crossValidateReductions[i] = m_reduction;
        }
        m_NumFolds = numFolds;
        return m_attributeSelections;
    }

    public Evaluation crossValidateModel(Classifier baseClassifier, Instances
            data, int numFolds, Random random) throws Exception {
        SemiRepeatFSEvaluation evaluation = new SemiRepeatFSEvaluation(data, m_labeledRatio);
        evaluation.setAttributeSelections(this.m_attributeSelections);
        evaluation.crossValidateModel(baseClassifier, data, numFolds, random);
        return evaluation;
    }
}
