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
 * Filter Feature Selection use this Evaluation. This evaluation can reuse the result of attribute selection on
 * different base classifier.
 * Created by xin on 11/06/2018.
 */
public class MultipleFSEvaluation {

    protected Instances m_data;

    protected Instances m_Header;

    /** The number of folds for a cross-validation. */
    protected int m_NumFolds;

    protected long m_seed;

    protected AttributeSelection[] m_attributeSelections;

    protected String[] m_crossValidateReductions;

    protected String m_reduction;

    protected double[] m_crossValidateTime;

    protected double m_time;

    protected int minFoldASLength;

    public MultipleFSEvaluation(Instances data){
        m_Header = new Instances(data, 0);
    }

    /**
     * Returns the header of the underlying dataset.
     *
     * @return the header information
     */
    public Instances getHeader() {
        return m_Header;
    }

    public int getMinFoldASLength() {
        return minFoldASLength;
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

    public void initCrossValidate(AttributeSelection attributeSelection, Instances data, int numFolds, long seed)
            throws Exception {
        this.m_NumFolds = numFolds;
        this.m_seed = seed;
        this.m_data = data;
        generateCrossFS(attributeSelection,m_data,m_NumFolds,new Random(seed));
        minFoldASLength = data.numAttributes()-1;
        for (AttributeSelection m_attributeSelection : m_attributeSelections) {
            if(m_attributeSelection.numberAttributesSelected()< minFoldASLength)
                minFoldASLength = m_attributeSelection.numberAttributesSelected();
        }
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

            AttributeSelection copiedAttributeSelection = AttributeSelection.makeCopy(attributeSelection);
            m_attributeSelections[i] = buildAttributeSelection(copiedAttributeSelection,train);
            m_reduction = m_attributeSelections[i].toResultsString();
            m_crossValidateTime[i] = m_time;
            m_crossValidateReductions[i] = m_reduction;
        }
        m_NumFolds = numFolds;
        return m_attributeSelections;
    }

    public AttributeSelection buildAttributeSelection(AttributeSelection m_AttributeSelection, Instances data) throws Exception {

        // get fresh Instances object
        Instances newData = new Instances(data);
        Instances resampledData = null;
        // check to see if training data has all equal weights
        double weight = newData.instance(0).weight();
        boolean ok = false;
        for (int i = 1; i < newData.numInstances(); i++) {
            if (newData.instance(i).weight() != weight) {
                ok = true;
                break;
            }
        }

        if (ok) {
            if (!(m_AttributeSelection.getFsAlgorithm() instanceof WeightedInstancesHandler)) {
                Random r = new Random(1);
                for (int i = 0; i < 10; i++) {
                    r.nextDouble();
                }
                resampledData = newData.resampleWithWeights(r);
            }
        } else {
            // all equal weights in the training data so just use as is
            resampledData = newData;
        }
        long start = System.currentTimeMillis();
        //if algorithm can handle weighted data just use it, otherwise use resampledData
        m_AttributeSelection.
                SelectAttributes((m_AttributeSelection.getFsAlgorithm() instanceof WeightedInstancesHandler)
                        ? newData
                        : resampledData);

        long end = System.currentTimeMillis();
        m_time = (double)(end - start);
        return m_AttributeSelection;
    }

    public Evaluation crossValidateModel(Classifier baseClassifier, Instances
            data, int numFolds, Random random) throws Exception {
        RepeatFSEvaluation evaluation = new RepeatFSEvaluation(data);
        evaluation.setAttributeSelections(this.m_attributeSelections);
        evaluation.crossValidateModel(baseClassifier,data,numFolds,random);
        return evaluation;
    }

    public Evaluation nextClassifier(Classifier baseClassifier) throws Exception {
        return crossValidateModel(baseClassifier,m_data,m_NumFolds,new Random(m_seed));
    }

    public Evaluation nextClassifier(Classifier baseClassifier,int count) throws Exception {
        return countCrossValidateModel(baseClassifier,count,m_data,m_NumFolds,new Random(m_seed));
    }

    public Evaluation countCrossValidateModel(Classifier baseClassifier, int featureCount, Instances
            data, int numFolds, Random random) throws Exception {
        AttributeSelection[] countAttributeSelection = new AttributeSelection[m_attributeSelections.length];
        for (int i = 0; i < m_attributeSelections.length; i++) {
            countAttributeSelection[i] = m_attributeSelections[i].generateAS(featureCount);
        }
        RepeatFSEvaluation evaluation = new RepeatFSEvaluation(data);
        evaluation.setAttributeSelections(countAttributeSelection);
        evaluation.crossValidateModel(baseClassifier,data,numFolds,random);
        return evaluation;
    }

    public double[] timeMeasures(){
        return m_crossValidateTime;
    }

    public double timeMeasure(){
        return Arrays.stream(m_crossValidateTime).reduce(Double::sum).orElse(0)/m_crossValidateTime.length;
    }
}
