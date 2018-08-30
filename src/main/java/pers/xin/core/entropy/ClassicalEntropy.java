package pers.xin.core.entropy;

import pers.xin.core.neighbor.Neighborhood;
import pers.xin.core.neighbor.Neighborhood2NORM;
import weka.core.ContingencyTables;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by xin on 06/03/2018.
 */
public class ClassicalEntropy implements Entropy{

    private Instances m_data;

    /**
     * 对一对简单属性的互信息缓存
     */
    double[][] MI_cache;

    double[] entropy_cache;

    public ClassicalEntropy(Instances data) {
        try {
            Discretize discretize = new Discretize();
            discretize.setInputFormat(data);
            m_data = Filter.useFilter(data,discretize);
        } catch (Exception e) {
            e.printStackTrace();
        }
        MI_cache = new double[m_data.numAttributes()][m_data.numAttributes()];
        entropy_cache = new double[m_data.numAttributes()];
        for (int i = 0; i < m_data.numAttributes(); i++) {
            entropy_cache[i] = entropy(i);
            for (int j = 0; j < m_data.numAttributes(); j++) {
                MI_cache[i][j]=-1;
            }
        }

    }

    public double mutualInformation(Set<Integer> a, Set<Integer> b){
        if(a==null||b==null||a.isEmpty()||b.isEmpty()) return 0;
        double ea = entropy(a);
        double eb = entropy(b);
        Set<Integer> ab = new HashSet<>(a);
        ab.addAll(b);
        double eab = entropy(ab);
        return ea + eb - eab;
    }

    public double mutualInformation(int a,int b){
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

    public double normalizedutualInformation(int a, int b){
        double ea = entropy_cache[a];
        double eb = entropy_cache[b];
        if(MI_cache[a][b]==-1){
            Set<Integer> ab = new HashSet<>();
            ab.add(a);
            ab.add(b);
            double eab = entropy(ab);
            MI_cache[a][b] = ea + eb - eab;
            MI_cache[b][a] = MI_cache[a][b];
        }
        if(MI_cache[a][b]==0) return 0;
        return MI_cache[a][b]/(ea<eb?ea:eb);
    }

    public double mutualInformation(Set<Integer> indicesR, int indexS){
        Set<Integer> s = new HashSet<>();
        s.add(indexS);
        double mi = mutualInformation(indicesR,s);
        return mi;
    }

    public double entropy(Set<Integer> s){
        Neighborhood nh = new Neighborhood2NORM(m_data,0,s);
        int numIns = m_data.numInstances();

        boolean[] flag = new boolean[numIns];
        ArrayList<Double> countList = new ArrayList<>();
        for (int i = 0; i < numIns; i++) {
            if(flag[i]==false){
                double c = 0;
                Instance first = m_data.get(i);
                for (int j = i; j < numIns; j++) {
                    if(nh.isNeighbor(first,m_data.get(j))){
                        c += 1;
                        flag[j] = true;
                    }
                }
                countList.add(c);
            }
        }
        double[] count = new double[countList.size()];
        for (int i = 0; i < countList.size(); i++) {
            count[i] = countList.get(i);
        }
        return ContingencyTables.entropy(count);
    }

    public double entropy(int index){
        Set<Integer> i = new HashSet<>();
        i.add(index);
        return entropy(i);
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
