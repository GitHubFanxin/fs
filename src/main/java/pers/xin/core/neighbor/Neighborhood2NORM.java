package pers.xin.core.neighbor;

import weka.core.Instance;
import weka.core.Instances;

import java.util.Set;

/**
 * Created by xin on 27/04/2018.
 */
public class Neighborhood2NORM extends Neighborhood{
    private double delta2=0;
    private double cutOff=0;

    public Neighborhood2NORM(Instances data, double delta, Set<Integer> indices) {
        super(data, delta, indices);
        delta2 = delta *delta;
        cutOff = delta * numActiveIndices;
    }

    public Neighborhood2NORM(Instances data, double delta) {
        super(data, delta);
        delta2 = delta *delta;
        cutOff = delta * numActiveIndices;
    }

    @Override
    protected double updateDistance(double currDist, double diff) {
        double result = currDist;
        result += diff*diff;
        return result;
    }

    /**
     * 两个对象是否相邻
     * @param first the first instance
     * @param second the second instance
     * @return isNeighbor
     */
    public boolean isNeighbor(Instance first, Instance second){
        if(first.equals(second)||first==second) return true;
        double distance2 = distance(first,second, cutOff, null);
        double normalizedDistance2 = distance2/numActiveIndices;
        return normalizedDistance2<=delta2?true:false;
    }


}
