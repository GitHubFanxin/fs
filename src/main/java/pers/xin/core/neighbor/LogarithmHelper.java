package pers.xin.core.neighbor;

/**
 * Created by xin on 2017/10/11.
 */
public class LogarithmHelper {

    /** The natural logarithm of 2 */
    public static final double log2 = Math.log(2);

    /** Cache of integer logs */
    private static final double MAX_INT_FOR_CACHE_PLUS_ONE = 10000;
    private static final double[] INT_LOG_N_CACHE = new double[(int)MAX_INT_FOR_CACHE_PLUS_ONE];

    /** Initialize cache */
    static {
        for (int i = 1; i < MAX_INT_FOR_CACHE_PLUS_ONE; i++) {
            double d = (double)i;
            INT_LOG_N_CACHE[i] = Math.log(d);
        }
    }

    /**
     * Help method for computing entropy.
     */
    public static double lnFunc(double num){

        if (num <= 0) {
            return 0;
        } else {

            // Use cache if we have a sufficiently small integer
            if (num < MAX_INT_FOR_CACHE_PLUS_ONE) {
                int n = (int)num;
                if ((double)n == num) {
                    return INT_LOG_N_CACHE[n];
                }
            }
            return Math.log(num);
        }
    }
}
