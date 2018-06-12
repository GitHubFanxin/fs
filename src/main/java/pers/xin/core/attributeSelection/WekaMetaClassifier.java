package pers.xin.core.attributeSelection;

import weka.classifiers.meta.AttributeSelectedClassifier;

/**
 * a meta classifier based on weka's AttributeSelectedClassifier
 * Created by xin on 10/06/2018.
 */
public class WekaMetaClassifier extends AttributeSelectedClassifier implements FSMeasure{

    @Override
    public String getReductionString() {
        StringBuffer sb = new StringBuffer();
        try {
            int[] reduction = m_AttributeSelection.selectedAttributes();

            for (int i = 0; i < reduction.length-1; i++) {
                if(i!=0) sb.append(",");
                sb.append(i);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return sb.toString();
    }
}
