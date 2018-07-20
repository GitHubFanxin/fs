package pers.xin.core.score;

import weka.core.Instances;

/**
 * Created by xin on 19/07/2018.
 */
public interface Scoring {

    void build(Instances dataset) throws Exception;

    double score(int attrIndex) throws Exception;

}
