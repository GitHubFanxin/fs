package pers.xin.core.optimization;

/**
 * Created by xin on 28/03/2018.
 */
public interface Function {

    /**
     * default interval is [0,1] and default precision is 2.
     */
    int dimension();

    /**
     * 计算适应度
     * @param params
     * @return
     */
    double computeFitness(Position params);
}
