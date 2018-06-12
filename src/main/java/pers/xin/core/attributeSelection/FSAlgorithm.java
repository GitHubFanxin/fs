package pers.xin.core.attributeSelection;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.SerializedObject;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * To extend this class must implement the select() method. Variable data is the train data.
 * Created by xin on 2017/4/13.
 */
public abstract class FSAlgorithm implements Serializable, OptionHandler{

    static final long serialVersionUID = 1;
    /**
     * Training data.
     */
    protected Instances data;

    protected ArrayList<Integer> numericIndices = new ArrayList<>();

    protected ArrayList<Integer> nominalIndices = new ArrayList<>();

    /**
     * Count of each class. Indices dependent on Weka's value Indices.
     */
    protected int[] classCounts;

    /**
     * holds the selected attributes include class index
     */
    protected int[] m_SelectedAttributes;

    /**
     * List of selected attributes
     */
    protected List<Integer> l_SelectedAttributes;

    /**
     * Implement algorithm here.
     * @return List contains attribute indices.
     */
    protected abstract List<Integer> select(Instances data) throws Exception;

    /**
     * Init some variables.
     */
    protected void init(Instances data){
        this.data = data;
        for (int i = 0; i < data.numAttributes()-1; i++) {
            if(data.attribute(i).isNumeric()) numericIndices.add(i);
            else nominalIndices.add(i);
        }
        classCounts = new int[data.numClasses()];
        data.stream().forEach(ins-> {
            if(!ins.classIsMissing()){
                classCounts[(int)ins.classValue()]++;
            }
        });
    }


    /**
     * Select attribute and save the attribute indices in m_SelectedAttributes include class index.
     * @param data The training data.
     */
    public int[] selectAttributes(Instances data) throws Exception{
        init(data);
        l_SelectedAttributes=select(this.data);
        m_SelectedAttributes=new int[l_SelectedAttributes.size()+1];
        for (int i = 0; i < l_SelectedAttributes.size(); i++) {
            m_SelectedAttributes[i]=l_SelectedAttributes.get(i);
        }
        m_SelectedAttributes[l_SelectedAttributes.size()]=data.classIndex();
        return m_SelectedAttributes;
    }

    public void reset(){
        this.m_SelectedAttributes=null;
    }

    /**
     * Get the list of selected attribute indices not include class index.
     * @return the list of selected.
     */
    public List<Integer> getReductionList(){
        return l_SelectedAttributes;
    }

    /**
     * Init the classCounts and return it.
     * @return Count of each class.
     */
    public int[] getClassCounts(){
        if(classCounts==null){
            classCounts = new int[data.numClasses()];
            data.stream().forEach(ins-> {
                if(!ins.classIsMissing()){
                    classCounts[(int)ins.classValue()]++;
                }
            });
        }
        return classCounts;
    }

    public void clean() {
        if (data != null) {
            // save memory
            data = new Instances(data, 0);
            numericIndices = null;
            nominalIndices = null;
            classCounts = null;
            m_SelectedAttributes = null;
            l_SelectedAttributes = null;
        }
    }

    public static FSAlgorithm makeCopy(FSAlgorithm model) throws Exception {

        return (FSAlgorithm) new SerializedObject(model).getObject();
    }

}
