package pers.xin.experiment;

import pers.xin.core.evaluation.FSEvaluation;
import pers.xin.core.evaluation.SemiFSEvaluation;
import pers.xin.utils.Output;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * Created by xin on 12/06/2018.
 */
public class SemiOriginExperiment extends OriginExperiment{
    protected double m_labelRatio;

    public SemiOriginExperiment(int numFold, long seed, double labelRatio) {
        super(numFold, seed);
        this.m_labelRatio = labelRatio;
    }

    /**
     * output result as table (data,classifier)-pct
     * @param file data file
     */
    public void oneDataMultipleClassifierExperiment(File file){
        try {
            FileInputStream fis = new FileInputStream(file);
            InputStreamReader reader = new InputStreamReader(fis,"UTF-8");
            Instances data = new Instances(reader);
            data.setClassIndex(data.numAttributes()-1);

            for (Classifier classifier : classifiers) {
                Evaluation evaluation = new SemiFSEvaluation(data,m_labelRatio);
                Classifier c = AbstractClassifier.makeCopy(classifier);
                evaluation.crossValidateModel(c,data,numFold,new Random(seed));
                PrintWriter pw = Output.createAppendPrint(classifier.getClass().getSimpleName());
                pw.println(evaluation.getHeader().relationName()+","+getMeasure(evaluation));
                pw.close();
            }

        } catch (Exception e){
            e.printStackTrace();
        }
    }

    public String getMeasure(Evaluation evaluation){
        return ""+evaluation.pctCorrect();
    }

    public static void main(String[] args) throws Exception {
        SemiOriginExperiment e = new SemiOriginExperiment(10,1,0.2);
        e.setDataFilePath("./dataset");
        J48 j48 = new J48();
        IBk iBk = new IBk();
        iBk.setKNN(3);
        Classifier[] cs = new Classifier[]{j48, iBk};
        e.setClassifiers(cs);
        e.setParallel(false);
        e.run();
    }
}
