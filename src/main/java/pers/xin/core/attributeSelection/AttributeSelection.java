package pers.xin.core.attributeSelection;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializedObject;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.Serializable;
import java.util.stream.Collectors;

/**
 * if algorithm can handle weighted data it should implement WeightedInstancesHandler
 * Created by xin on 23/03/2018.
 */
public class AttributeSelection implements Serializable {

    static final long serialVersionUID = 1;

    /** the instances to select attributes from */
    protected Instances m_trainInstances;

    /** holds a string describing the results of the attribute selection */
    private final StringBuffer m_selectionResults;

    /**
     * the attribute filter for processing instances with respect to the most
     * recent feature selection run
     */
    protected Remove m_attributeFilter = null;

    /** the selected attributes */
    protected int[] m_selectedAttributeSet;

    protected FSAlgorithm fsAlgorithm = null;

    public AttributeSelection() {
        m_selectionResults = new StringBuffer();
        m_selectedAttributeSet = null;
    }

    public AttributeSelection(FSAlgorithm fsAlgorithm) {
        this.fsAlgorithm = fsAlgorithm;
        m_selectionResults = new StringBuffer();
        m_selectedAttributeSet = null;
    }

    public FSAlgorithm getFsAlgorithm() {
        return fsAlgorithm;
    }

    public boolean isBuilt(){
        return m_attributeFilter==null?false:true;
    }

    /**
     * Return the number of attributes selected from the most recent run of
     * attribute selection
     * @return number of attributes selected
     * @throws Exception if attribute selection has not been performed yet
     */
    public int numberAttributesSelected() throws Exception {
        int[] att = selectedAttributes();
        return att.length - 1;
    }

    /**
     * get the final selected set of attributes.
     *
     * @return an array of attribute indexes
     * @exception Exception if attribute selection has not been performed yet
     */
    public int[] selectedAttributes() throws Exception {
        if (m_selectedAttributeSet == null) {
            throw new Exception("Attribute selection has not been performed yet!");
        }
        return m_selectedAttributeSet;
    }

    /**
     * get a description of the attribute selection
     *
     * @return a String describing the results of attribute selection
     */
    public String toResultsString() {
        return m_selectionResults.toString();
    }

    /**
     * reduce the dimensionality of a set of instances to include only those
     * attributes chosen by the last run of attribute selection.
     *
     * @param in the instances to be reduced
     * @return a dimensionality reduced set of instances
     * @exception Exception if the instances can't be reduced
     */
    public Instances reduceDimensionality(Instances in) throws Exception {
        if (m_attributeFilter == null) {
            throw new Exception("No feature selection has been performed yet!");
        }
        return Filter.useFilter(in, m_attributeFilter);
    }

    /**
     * reduce the dimensionality of a single instance to include only those
     * attributes chosen by the last run of attribute selection.
     *
     * @param in the instance to be reduced
     * @return a dimensionality reduced instance
     * @exception Exception if the instance can't be reduced
     */
    public Instance reduceDimensionality(Instance in) throws Exception {
        if (m_attributeFilter == null) {
            throw new Exception("No feature selection has been performed yet!");
        }
        m_attributeFilter.input(in);
        m_attributeFilter.batchFinished();
        Instance result = m_attributeFilter.output();
        return result;
    }

    /**
     * Perform attribute selection on the supplied training instances.
     *
     * @param data the instances to select attributes from
     * @exception Exception if there is a problem during selection
     */
    public void SelectAttributes(Instances data) throws Exception {
        m_attributeFilter = null;
        m_trainInstances = data;

        if(fsAlgorithm == null){
            throw new Exception("No feature selection algorithm has been set yet!");
        }

        m_selectedAttributeSet = fsAlgorithm.selectAttributes(data);

        m_selectionResults.append(
                fsAlgorithm.getReductionList().stream().map(i->String.valueOf(i+1)).collect(Collectors.joining(",")));

        // set up the attribute filter with the selected attributes
        if (m_selectedAttributeSet != null) {
            m_attributeFilter = new Remove();
            m_attributeFilter.setAttributeIndicesArray(m_selectedAttributeSet);
            m_attributeFilter.setInvertSelection(true);
            m_attributeFilter.setInputFormat(m_trainInstances);
        }

        fsAlgorithm.clean();
    }

    @Override
    public String toString() {
        return m_selectionResults.toString();
    }

    public static AttributeSelection makeCopy(AttributeSelection model) throws Exception {

        return (AttributeSelection) new SerializedObject(model).getObject();
    }

    public AttributeSelection generateAS(int count) throws Exception {
        int[] countSelectedAttributeSet = new int[count+1];
        for (int i = 0; i < count; i++) {
            countSelectedAttributeSet[i] = m_selectedAttributeSet[i];
        }
        countSelectedAttributeSet[count] = m_selectedAttributeSet[m_selectedAttributeSet.length-1];
        AttributeSelection copy = makeCopy(this);
        copy.m_selectedAttributeSet = countSelectedAttributeSet;
        if (copy.m_selectedAttributeSet != null) {
            copy.m_attributeFilter = new Remove();
            copy.m_attributeFilter.setAttributeIndicesArray(countSelectedAttributeSet);
            copy.m_attributeFilter.setInvertSelection(true);
            copy.m_attributeFilter.setInputFormat(m_trainInstances);
        }
        return copy;
    }
}
