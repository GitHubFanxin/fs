package pers.xin.core.entropy;

import java.util.Set;

/**
 * Created by xin on 27/04/2018.
 */
public interface Entropy {
    double mutualInformation(Set<Integer> a, Set<Integer> b);

    double mutualInformation(int indexR, int indexS);

    double mutualInformation(Set<Integer> indicesR, int indexS);

    double entropy(Set<Integer> s);

    double SymmetricalUncertainty(Set<Integer> a, Set<Integer> b);

    double SymmetricalUncertainty(Set<Integer> a, int b);

    double SymmetricalUncertainty(int a, int b);
}
