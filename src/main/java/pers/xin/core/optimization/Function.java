package pers.xin.core.optimization;

/**
 * Created by xin on 28/03/2018.
 */
public interface Function {

    /**
     * Param dimension of the function.
     */
    int dimension();

    /**
     * Method can compute fitness. Pay attention to that search interval is [0,1]. You can multiply some coefficients
     * to make the search interval become [0,1]
     * @param params
     * @return
     */
    double computeFitness(Position params);
}
