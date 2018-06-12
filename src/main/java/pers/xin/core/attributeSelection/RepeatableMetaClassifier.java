package pers.xin.core.attributeSelection;

import weka.core.Instances;
import weka.core.WeightedInstancesHandler;

import java.util.Random;

/**
 * a classifier that can use a attribute selection which has already selected attributes.
 * Created by xin on 11/06/2018.
 */
public class RepeatableMetaClassifier extends MetaClassifier {

    /**
     * Build the classifier on the dimensionally reduced data.
     *
     * @param data the training data
     * @throws Exception if the classifier could not be built successfully
     */
    @Override
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

        if(!m_AttributeSelection.isBuilt()){
            //if algorithm can handle weighted data just use it, otherwise use resampledData
            m_AttributeSelection.
                    SelectAttributes((m_AttributeSelection.getFsAlgorithm() instanceof WeightedInstancesHandler)
                            ? newData
                            : resampledData);
        }


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
}
