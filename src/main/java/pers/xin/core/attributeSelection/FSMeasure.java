package pers.xin.core.attributeSelection;

/**
 * Created by xin on 10/06/2018.
 */
public interface FSMeasure {
    /**
     * Additional measure --- number of attributes selected
     * @return the number of attributes selected
     */
    double measureNumAttributesSelected();

    /**
     * Additional measure --- time taken (milliseconds) to select the attributes
     * @return the time taken to select attributes
     */
    double measureSelectionTime();

    /**
     * Additional measure --- time taken (milliseconds) to select attributes
     * and build the classifier
     * @return the total time (select attributes + build classifier)
     */
    double measureTime();

    /**
     * Returns feature selection result start with 1 not include class
     *
     * @return feature selection result
     */
    String getReductionString();

}
