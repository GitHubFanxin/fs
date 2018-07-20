package pers.xin.core.score;

import pers.xin.core.score.utils.NominalToBinary;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

import java.io.File;
import java.io.FileReader;

/**
 * Created by xin on 19/07/2018.
 */
public class FisherScore implements Scoring {
    protected double[] scores;


    private double[] m_build(Instances data) {
        int numAttr = data.numAttributes() - 1;
        double[] m_scores = new double[numAttr];

        double[] attrSum = new double[numAttr];
        double[][] attrSumInClass = new double[data.numClasses()][numAttr];
        double[][] attrSumSqInClass = new double[data.numClasses()][numAttr];
        double[] classCount = new double[data.numClasses()];

        for (Instance datum : data) {
            for (int d = 0; d < numAttr; d++) {
                double x = datum.value(d);
                if (Double.isNaN(x))
                    continue;//pass the miss value
                attrSum[d] += x;
                attrSumInClass[(int) datum.classValue()][d] += x;
                attrSumSqInClass[(int) datum.classValue()][d] += x * x;
            }
            classCount[(int) datum.classValue()] += 1;
        }
        double[] attrMean = new double[numAttr];
        for (int d = 0; d < numAttr; d++) {
            attrMean[d] = attrSum[d]/data.numInstances();
        }

        for (int d = 0; d < numAttr; d++) {
            double Sb = 0;
            double Sw = 0;
            for (int i = 0; i < data.numClasses(); i++) {
                double var = (attrSumSqInClass[i][d] - attrSumInClass[i][d] * attrSumInClass[i][d] / classCount[i]) /
                        (classCount[i] - 1);
                double mean = attrSumInClass[i][d] / classCount[i];
                Sb += classCount[i]*(mean - attrMean[d])*(mean - attrMean[d]);
                Sw += classCount[i]*var*var;
            }
            m_scores[d] = Sb/Sw;
        }
        return m_scores;
    }

    @Override
    public void build(Instances dataset) throws Exception {
        scores = new double[dataset.numAttributes()-1];
        NominalToBinary ntb = new NominalToBinary();
        ntb.setInputFormat(dataset);
        Instances newdata = Filter.useFilter(dataset,ntb);
        int[][] map = ntb.getMap();
        for (int a = 0; a < map.length; a++) {
            for (int i = map[a][0];i<map[a][1]; i++){
                System.out.println(newdata.attribute(i));
            }
            System.out.println("-----------------");
        }
        double[] m_score = m_build(newdata);
        for (int d = 0; d < dataset.numAttributes()-1; d++) {
            double tmpScore = 0;
            for (int i = map[d][0];i<map[d][1]; i++){
                tmpScore+=m_score[i];
            }
            tmpScore = tmpScore / (map[d][1] - map[d][0]);
            if(Double.isNaN(tmpScore)) scores[d] = 0;
            else scores[d] = tmpScore;
        }
    }

    @Override
    public double score(int attrIndex) throws Exception {
        if (scores == null) throw new Exception("model has not been built!");
        return scores[attrIndex];
    }

    public static void main(String[] args) throws Exception {
        File file = new File("./dataset/sick.arff");
        Instances data = new Instances(new FileReader((file)));
        data.setClassIndex(data.numAttributes()-1);
        FisherScore fs = new FisherScore();
        fs.build(data);
        for (int i = 0; i < data.numAttributes()-1; i++) {
            System.out.println(i+"\t"+fs.score(i));
        }
    }
}
