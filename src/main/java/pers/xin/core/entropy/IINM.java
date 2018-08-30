package pers.xin.core.entropy;

import pers.xin.core.neighbor.LogarithmHelper;
import pers.xin.core.neighbor.Neighborhood;
import pers.xin.core.neighbor.NeighborhoodInfinity;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashSet;
import java.util.Set;

/**
 * Created by xin on 03/03/2018.
 */
public class IINM implements Entropy{
    /** The natural logarithm of 2 */
    public static final double log2 = Math.log(2);

    /**
     * Help method for computing entropy.
     */
    private double lnFunc(double num){
        return LogarithmHelper.lnFunc(num);
    }

    /**
     * delta for neighborhood rough set.
     */
    private double delta;

    /**
     * 计算熵的数据集
     */
    private Instances m_data;

    /**
     * 用于归一化得数据集
     */
    private Instances normalize_data;

    /**
     * 对一对简单属性的互信息缓存
     */
    double[][] MI_cache;

    double[] entropy_cache;

    public IINM(double delta, Instances data) {
        this.delta = delta;
        this.m_data = data;
        MI_cache = new double[m_data.numAttributes()][m_data.numAttributes()];
        entropy_cache  = new double[m_data.numAttributes()];
        this.normalize_data = data;
        for (int i = 0; i < m_data.numAttributes(); i++) {
            entropy_cache[i] = entropy(i);
            for (int j = 0; j < m_data.numAttributes(); j++) {
                MI_cache[i][j]=-1;
            }
        }
    }

    public IINM(double delta, Instances data, Instances normalize_data) {
        this(delta,data);
        this.normalize_data = normalize_data;
    }

    public double entropy(Set<Integer> indices){
        int num = m_data.numInstances();
        double returnValue = 0;
        Neighborhood neighborhoodHelper = new NeighborhoodInfinity(normalize_data,delta,indices);
        int[] neighborCount = new int[num];
        for (int i = 0; i < num; i++) {
            int numNeighbor = 0;
            for (int j = i; j < num; j++) {
                //                if(i==j) continue;
                Instance first = m_data.get(i);
                Instance second = m_data.get(j);
                if(neighborhoodHelper.isNeighbor(first,second)){
                    neighborCount[i]++;
                    neighborCount[j]++;
                }
//                numNeighbor += neighborhoodHelper.isNeighbor(first,second) ? 1 : 0;
            }
//            returnValue += lnFunc(numNeighbor);
        }
        for (int i = 0; i < num; i++) returnValue += lnFunc(neighborCount[i]);
        returnValue -= num*lnFunc(num);
        returnValue = returnValue /(log2*num);
        return -returnValue;
    }

    public double entropy(int index){
        Set<Integer> i = new HashSet<>();
        i.add(index);
        return entropy(i);
    }

    /**
     * NH(R|S)
     * @param indicesR attribute set R
     * @param indicesS attribute set S
     * @return conditional entropy
     */
    public double conditionalEntropy(Set<Integer> indicesR, Set<Integer> indicesS){
        int num = m_data.numInstances();
        double returnValue = 0;
        Set<Integer> indicesRUnionS = new HashSet<>();
        indicesRUnionS.addAll(indicesR);
        indicesRUnionS.addAll(indicesS);
        Neighborhood neighborhoodHelperRJointS = new NeighborhoodInfinity(normalize_data,delta,
                indicesRUnionS);
        Neighborhood neighborhoodHelperS = new NeighborhoodInfinity(normalize_data,delta,indicesS);
        for (int i = 0; i < num; i++) {
            int numNeighborRJointS = 0;
            int numNeighborS = 0;
            Instance first = m_data.get(i);
            for (int j = 0; j < num; j++) {
//                if(i==j) continue;
                Instance second = m_data.get(j);
                numNeighborRJointS += neighborhoodHelperRJointS.isNeighbor(first, second) ? 1 : 0;
                numNeighborS += neighborhoodHelperS.isNeighbor(first, second) ? 1 : 0;
            }
            returnValue += lnFunc(numNeighborRJointS)-lnFunc(numNeighborS);
        }
        returnValue = returnValue/(log2*num);
        return -returnValue;
    }

    public double conditionalEntropy(int indexR, int indexS){
        Set<Integer> r = new HashSet<>();
        Set<Integer> s = new HashSet<>();
        r.add(indexR);
        s.add(indexS);
        return conditionalEntropy(r,s);
    }

    /**
     * NMI(R;S)
     * @param indicesR attribute set R
     * @param indicesS attribute set S
     * @return NMI
     */
    public double mutualInformation(Set<Integer> indicesR, Set<Integer> indicesS){
        if(indicesR==null||indicesS==null||indicesR.isEmpty()||indicesS.isEmpty())
            return 0;
        int num = m_data.numInstances();
        double returnValue = 0;
        Set<Integer> indicesRUnionS = new HashSet<>();
        indicesRUnionS.addAll(indicesR);
        indicesRUnionS.addAll(indicesS);
        Neighborhood neighborhoodHelperRJointS = new NeighborhoodInfinity(normalize_data,delta,indicesRUnionS);

        Neighborhood neighborhoodHelperR = new NeighborhoodInfinity(normalize_data,delta,indicesR);

        Neighborhood neighborhoodHelperS = new NeighborhoodInfinity(normalize_data,delta,indicesS);

        for (int i = 0; i < num; i++) {
            int numNeighborRJointS = 0;
            int numNeighborR = 0;
            int numNeighborS = 0;
            Instance first = m_data.get(i);
            for (int j = 0; j < num; j++) {
//                if(i==j) continue;
                Instance second = m_data.get(j);
                numNeighborRJointS += neighborhoodHelperRJointS.isNeighbor(first, second) ? 1 : 0;
                numNeighborS += neighborhoodHelperS.isNeighbor(first, second) ? 1 : 0;
                numNeighborR += neighborhoodHelperR.isNeighbor(first, second) ? 1 : 0;
            }
            returnValue += lnFunc(numNeighborR) + lnFunc(numNeighborS) - lnFunc(num) -
                    lnFunc(numNeighborRJointS);
        }
        returnValue = returnValue/(log2*num);
        return -returnValue;
    }

    public double mutualInformation(Set<Integer> indicesR, int indexS){
        Set<Integer> s = new HashSet<>();
        s.add(indexS);
        double mi = mutualInformation(indicesR,s);
        return mi;
    }

//    public double mutualInformation(int indexR, int indexS){
//        if(MI_cache[indexR][indexS]==0){
//            Set<Integer> r = new HashSet<>();
//            Set<Integer> s = new HashSet<>();
//            r.add(indexR);
//            s.add(indexS);
//            double mi = mutualInformation(r,s);
//            MI_cache[indexR][indexS] = mi;
//            MI_cache[indexS][indexR] = mi;
//        }
//        return MI_cache[indexR][indexS];
//    }

    public double mutualInformation(int a, int b){
        if(MI_cache[a][b]==-1){
            double ea = entropy_cache[a];
            double eb = entropy_cache[b];
            Set<Integer> ab = new HashSet<>();
            ab.add(a);
            ab.add(b);
            double eab = entropy(ab);
            MI_cache[a][b] = ea + eb - eab;
            MI_cache[b][a] = MI_cache[a][b];
        }
        return MI_cache[a][b];
    }


    public double jointEntropy(Set<Integer> indicesR, Set<Integer> indicesS){
        int num = m_data.numInstances();
        double returnValue = 0;
        Set<Integer> indicesRUnionS = new HashSet<>();
        indicesRUnionS.addAll(indicesR);
        indicesRUnionS.addAll(indicesS);
        Neighborhood neighborhoodHelperRJointS = new NeighborhoodInfinity(normalize_data,delta,indicesRUnionS);
        for (int i = 0; i < num; i++) {
            int numNeighborRJointS = 0;
            Instance first = m_data.get(i);
            for (int j = 0; j < num; j++) {
//                if(i==j) continue;
                Instance second = m_data.get(j);
                numNeighborRJointS += neighborhoodHelperRJointS.isNeighbor(first, second) ? 1 : 0;
            }
            returnValue += lnFunc(numNeighborRJointS);
        }
        returnValue -= num*lnFunc(num);
        returnValue = returnValue/(log2*num);
        return -returnValue;
    }

    public double jointEntropy(int indexR, int indexS){
        Set<Integer> r = new HashSet<>();
        Set<Integer> s = new HashSet<>();
        r.add(indexR);
        s.add(indexS);
        return jointEntropy(r,s);
    }

    @Override
    public double SymmetricalUncertainty(Set<Integer> a, Set<Integer> b) {
        double mi = mutualInformation(a,b);
        if(mi<=0) return 0;
        double ha = entropy(a);
        double hb = entropy(b);
        return 2*mi/(ha*hb);
    }

    @Override
    public double SymmetricalUncertainty(Set<Integer> a, int b) {
        Set<Integer> B = new HashSet<>();
        B.add(b);
        return 0;
    }

    @Override
    public double SymmetricalUncertainty(int a, int b) {
        double mi = mutualInformation(a,b);
        if(mi<=0) return 0;
        double ha = entropy(a);
        double hb = entropy(b);
        return 2*mi/(ha*hb);
    }
}
