package pers.xin.core.attributeSelection;


import weka.classifiers.SingleClassifierEnhancer;
import weka.core.*;
import weka.core.Capabilities.Capability;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

/**
 <!-- globalinfo-start -->
 * Dimensionality of training and test data is reduced by attribute selection before being passed on to a classifier.
 * <dimension/>
 <!-- globalinfo-end -->
 *
 <!-- options-start -->
 * Valid options are: <dimension/>
 *
 <!-- options-end -->
 */
public class MetaClassifier
        extends SingleClassifierEnhancer
        implements OptionHandler, Drawable, AdditionalMeasureProducer,
        WeightedInstancesHandler,FSMeasure {

    /** for serialization */
    static final long serialVersionUID = -1L;

    /** The attribute selection object */
    protected AttributeSelection m_AttributeSelection = null;


    /** The header of the dimensionally reduced data */
    protected Instances m_ReducedHeader;

    /** The number of class vals in the training data (1 if class is numeric) */
    protected int m_numClasses;

    /** The number of attributes selected by the attribute selection phase */
    protected double m_numAttributesSelected;

    /** The time taken to select attributes in milliseconds */
    protected double m_selectionTime;

    /** The time taken to select attributes AND build the classifier */
    protected double m_totalTime;


    /**
     * String describing default classifier.
     *
     * @return the default classifier classname
     */
    protected String defaultClassifierString() {

        return "weka.classifiers.trees.J48";
    }

    /**
     * Default constructor(J48).
     */
    public MetaClassifier() {
        m_Classifier = new weka.classifiers.trees.J48();
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return      the capabilities of this classifier
     */
    public Capabilities getCapabilities() {
        Capabilities	result;

        result = super.getCapabilities();


        // set dependencies
        for (Capability cap: Capability.values())
            result.enableDependency(cap);

        return result;
    }

    /**
     * write by xin
     * set attributeSelection
     * @param m_AttributeSelection
     */
    public void setAttributeSelection(AttributeSelection m_AttributeSelection) {
        this.m_AttributeSelection = m_AttributeSelection;
    }

    public AttributeSelection getM_AttributeSelection() {
        return m_AttributeSelection;
    }

    /**
     * Build the classifier on the dimensionally reduced data.
     *
     * @param data the training data
     * @throws Exception if the classifier could not be built successfully
     */
    public void buildClassifier(Instances data) throws Exception {
        if (m_Classifier == null) {
            throw new Exception("No base classifier has been set!");
        }


        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        // get fresh Instances object
        Instances newData = new Instances(data);

        if (newData.numInstances() == 0) {
            m_Classifier.buildClassifier(newData);
            return;
        }
        if (newData.classAttribute().isNominal()) {
            m_numClasses = newData.classAttribute().numValues();
        } else {
            m_numClasses = 1;
        }

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

        if(m_AttributeSelection==null){
            throw new Exception("No feature selection has been set yet!");
        }

        if (ok) {
            if (!(m_AttributeSelection.getFsAlgorithm() instanceof WeightedInstancesHandler) ||
                    !(m_Classifier instanceof WeightedInstancesHandler)) {
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

        if (m_Classifier instanceof WeightedInstancesHandler) {
            newData = m_AttributeSelection.reduceDimensionality(newData);
            m_Classifier.buildClassifier(newData);
        } else {
            resampledData = m_AttributeSelection.reduceDimensionality(resampledData);
            m_Classifier.buildClassifier(resampledData);
        }

        long end2 = System.currentTimeMillis();

        m_numAttributesSelected = m_AttributeSelection.numberAttributesSelected();
        m_ReducedHeader =
                new Instances((m_Classifier instanceof WeightedInstancesHandler) ?
                        newData
                        : resampledData, 0);
        m_selectionTime = (double)(end - start);
        m_totalTime = (double)(end2 - start);
    }

    /**
     * Classifies a given instance after attribute selection
     *
     * @param instance the instance to be classified
     * @return the class distribution
     * @throws Exception if instance could not be classified
     * successfully
     */
    public double [] distributionForInstance(Instance instance)
            throws Exception {

        Instance newInstance;
        if (m_AttributeSelection == null) {
            //      throw new Exception("MetaClassifier: No model built yet!");
            newInstance = instance;
        } else {
            newInstance = m_AttributeSelection.reduceDimensionality(instance);
        }

        return m_Classifier.distributionForInstance(newInstance);
    }

    /**
     *  Returns the type of graph this classifier
     *  represents.
     *
     *  @return the type of graph
     */
    public int graphType() {

        if (m_Classifier instanceof Drawable)
            return ((Drawable)m_Classifier).graphType();
        else
            return Drawable.NOT_DRAWABLE;
    }

    /**
     * Returns graph describing the classifier (if possible).
     *
     * @return the graph of the classifier in dotty format
     * @throws Exception if the classifier cannot be graphed
     */
    public String graph() throws Exception {

        if (m_Classifier instanceof Drawable)
            return ((Drawable)m_Classifier).graph();
        else throw new Exception("Classifier: " + getClassifierSpec()
                + " cannot be graphed");
    }

    /**
     * Returns feature selection result start with 1 not include class
     *
     * @return feature selection result
     */
    public String getReductionString(){
        return m_AttributeSelection.toResultsString();
    }


    /**
     * Output a representation of this classifier
     *
     * @return a representation of this classifier
     */
    public String toString() {
        if (m_AttributeSelection == null) {
            return "MetaClassifier: No attribute selection possible.\n\n"
                    +m_Classifier.toString();
        }

        StringBuffer result = new StringBuffer();
        result.append("MetaClassifier:\n\n");
        result.append(m_AttributeSelection.toResultsString());
        result.append("\n\nHeader of reduced data:\n"+m_ReducedHeader.toString());
        result.append("\n\nClassifier Model\n"+m_Classifier.toString());

        return result.toString();
    }

    /**
     * Additional measure --- number of attributes selected
     * @return the number of attributes selected
     */
    public double measureNumAttributesSelected() {
        return m_numAttributesSelected;
    }

    /**
     * Additional measure --- time taken (milliseconds) to select the attributes
     * @return the time taken to select attributes
     */
    public double measureSelectionTime() {
        return m_selectionTime;
    }

    /**
     * Additional measure --- time taken (milliseconds) to select attributes
     * and build the classifier
     * @return the total time (select attributes + build classifier)
     */
    public double measureTime() {
        return m_totalTime;
    }

    /**
     * Returns an enumeration of the additional measure names
     * @return an enumeration of the measure names
     */
    public Enumeration<String> enumerateMeasures() {
        Vector<String> newVector = new Vector<String>(3);
        newVector.addElement("measureNumAttributesSelected");
        newVector.addElement("measureSelectionTime");
        newVector.addElement("measureTime");
        if (m_Classifier instanceof AdditionalMeasureProducer) {
            newVector.addAll(Collections.list(((AdditionalMeasureProducer)m_Classifier).
                    enumerateMeasures()));
        }
        return newVector.elements();
    }

    /**
     * Returns the value of the named measure
     * @param additionalMeasureName the name of the measure to query for its value
     * @return the value of the named measure
     * @throws IllegalArgumentException if the named measure is not supported
     */
    public double getMeasure(String additionalMeasureName) {
        if (additionalMeasureName.compareToIgnoreCase("measureNumAttributesSelected") == 0) {
            return measureNumAttributesSelected();
        } else if (additionalMeasureName.compareToIgnoreCase("measureSelectionTime") == 0) {
            return measureSelectionTime();
        } else if (additionalMeasureName.compareToIgnoreCase("measureTime") == 0) {
            return measureTime();
        } else if (m_Classifier instanceof AdditionalMeasureProducer) {
            return ((AdditionalMeasureProducer)m_Classifier).
                    getMeasure(additionalMeasureName);
        } else {
            throw new IllegalArgumentException(additionalMeasureName
                    + " not supported (MetaClassifier)");
        }
    }

    /**
     * Returns the revision string.
     *
     * @return		the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 11461 $");
    }

    /**
     * TestO method for testing this class.
     *
     * @param argv should contain the following arguments:
     * -t training file [-T test file] [-c class index]
     */
    public static void main(String [] argv) {
        runClassifier(new MetaClassifier(), argv);
    }
}
