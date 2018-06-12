package pers.xin.core.neighbor;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.NormalizableDistance;
import weka.core.Utils;
import weka.core.neighboursearch.PerformanceStats;

import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * 默认激活除了class的属性
 * Created by xin on 03/03/2018.
 */
public abstract class Neighborhood extends NormalizableDistance {

    protected double delta;

    protected int numActiveIndices=0;

    /**
     * 邻域计算帮助类
     * @param data 用于数据归一化的数据
     * @param delta 邻域大小
     * @param indices 参与计算的属性集合
     */
    public Neighborhood(Instances data, double delta, Set<Integer> indices) {
        super(data);
        this.delta = delta;
        setAttributeIndices(indices.stream().map(x->(x+1)+"").collect(Collectors.joining(",")));
        numActiveIndices = indices.size();
    }

    /**
     * 邻域计算帮助类，默认所有条件属性参与计算
     * @param data 用于数据归一化的数据
     * @param delta 邻域大小
     */
    public Neighborhood(Instances data, double delta) {
        super(data);
        this.delta = delta;
        numActiveIndices = m_Data.numAttributes()-1;
        Set<Integer> indices = new HashSet<>();
        for (int i = 0; i < data.numAttributes()-1; i++) indices.add(i);
        setAttributeIndices(indices.stream().map(x->(x+1)+"").collect(Collectors.joining(",")));
        numActiveIndices = indices.size();
    }

    @Override
    public String globalInfo() {
        return null;
    }

    @Override
    public String getRevision() {
        return null;
    }

    /**
     * 参与邻域计算的属性
     * @param indices 属性集合
     */
    public void setIndices(Set<Integer> indices){
        setAttributeIndices(indices.stream().map(x->(x+1)+"").collect(Collectors.joining(",")));
        numActiveIndices = indices.size();
    }

    /**
     * 由于原API中不能计算class类的距离在此重写删除该限制
     * @param first
     * @param second
     * @param cutOffValue
     * @param stats
     * @return
     */
    @Override
    public double distance(Instance first, Instance second, double cutOffValue,
                           PerformanceStats stats) {
        double distance = 0;
        int firstI, secondI;
        int firstNumValues = first.numValues();
        int secondNumValues = second.numValues();
        int numAttributes = m_Data.numAttributes();

        validate();

        for (int p1 = 0, p2 = 0; p1 < firstNumValues || p2 < secondNumValues;) {
            if (p1 >= firstNumValues) {
                firstI = numAttributes;
            } else {
                firstI = first.index(p1);
            }

            if (p2 >= secondNumValues) {
                secondI = numAttributes;
            } else {
                secondI = second.index(p2);
            }

            if ((firstI < numAttributes) && !m_ActiveIndices[firstI]) {
                p1++;
                continue;
            }

            if ((secondI < numAttributes) && !m_ActiveIndices[secondI]) {
                p2++;
                continue;
            }

            double diff;

            if (firstI == secondI) {
                diff = difference(firstI, first.valueSparse(p1), second.valueSparse(p2));
                p1++;
                p2++;
            } else if (firstI > secondI) {
                diff = difference(secondI, 0, second.valueSparse(p2));
                p2++;
            } else {
                diff = difference(firstI, first.valueSparse(p1), 0);
                p1++;
            }
            if (stats != null) {
                stats.incrCoordCount();
            }

            distance = updateDistance(distance, diff);
            if (distance > cutOffValue) {
                return Double.POSITIVE_INFINITY;
            }
        }

        return distance;
    }

    /**
     * 如果两个值都miss认为它们相似，即认为其没有分辨力
     */
    @Override
    protected double difference(int index, double val1, double val2) {
        if (Utils.isMissingValue(val1) || Utils.isMissingValue(val2)) return 0;
        return super.difference(index, val1, val2);
    }

    /**
     * 两个对象指定的某一属性上是否相邻
     * @param index
     * @param first
     * @param second
     * @return
     */
    public boolean isNeighborOnAttr(int index, Instance first, Instance second){
        if(first.equals(second)||first==second) return true;
//        if(first.isMissing(index)||second.isMissing(index)) return true;
        double dis =Math.abs(difference(index,first.value(index),second.value(index)));
        return dis<=delta;
    }

    /**
     * 两个对象是否相邻
     * @param first
     * @param second
     * @return
     */
    public abstract boolean isNeighbor(Instance first, Instance second);
}
