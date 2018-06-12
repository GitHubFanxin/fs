package pers.xin.core.neighbor;

import weka.core.Instance;
import weka.core.Instances;

import java.util.Set;

/**
 * Infinity norm
 * Created by xin on 05/04/2018.
 */
public class NeighborhoodInfinity extends Neighborhood {

    public NeighborhoodInfinity(Instances data, double delta, Set<Integer> indices) {
        super(data, delta, indices);
    }

    public NeighborhoodInfinity(Instances data, double delta) {
        super(data, delta);
    }

    @Override
    protected double updateDistance(double currDist, double diff) {
        double absDiff = Math.abs(diff);
        double result = absDiff>currDist?absDiff:currDist;
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
        double distance = distance(first,second, delta, null);
        return distance <= delta?true:false;
    }
}
