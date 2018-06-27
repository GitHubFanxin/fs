package pers.xin.experiment;

import pers.xin.algorithm.TEST;
import pers.xin.core.attributeSelection.AttributeSelection;
import pers.xin.core.attributeSelection.FSAlgorithm;
import pers.xin.core.evaluation.MultipleFSEvaluation;
import pers.xin.core.evaluation.SemiMultipleFSEvaluation;
import pers.xin.utils.Output;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * run with multiple classifier without selecting feature repeatedly.
 * Created by xin on 11/06/2018.
 */
public class SemiRepeatableExperiment extends RepeatableExperiment{
    protected double m_labelRatio;

    public SemiRepeatableExperiment(int numFold, long seed, double labelRatio) {
        super(numFold,seed);
        m_labelRatio = labelRatio;
    }

    /**
     * output result as table (data,classifier)-pct
     * @param file data file
     * @param outputMode output mode
     */
    public void oneDataMultipleClassifierExperiment(File file, int outputMode){
        try {
            FileInputStream fis = new FileInputStream(file);
            InputStreamReader reader = new InputStreamReader(fis,"UTF-8");
            Instances data = new Instances(reader);

//            Instances data = new Instances(new FileReader(file));
            data.setClassIndex(data.numAttributes()-1);
            FSAlgorithm copiedAlgorithm = FSAlgorithm.makeCopy(m_fsAlgorithm);
            AttributeSelection as = new AttributeSelection(copiedAlgorithm);
            SemiMultipleFSEvaluation multipleFSEvaluation = new SemiMultipleFSEvaluation(data,m_labelRatio);
            multipleFSEvaluation.initCrossValidate(as,data,numFold,seed);
            String[] reductions = multipleFSEvaluation.getCrossValidateReductions();

            switch (outputMode){
                case DATA_CLASSIFIER:
                    dataClassifier(multipleFSEvaluation,optionString);
                    break;
                case DATA_CLASSIFIERS:
                    dataClassifiers(multipleFSEvaluation,optionString);
                    break;
                case DATA_BOTH:
                    dataBoth(multipleFSEvaluation,optionString);
                    break;
                case COUNT_CLASSIFIER:
                    countClassifier(multipleFSEvaluation,optionString);
                    break;
                case COUNT_CLASSIFIERS:
                    countClassifiers(multipleFSEvaluation,optionString);
                    break;
                case COUNT_BOTH:
                    countBoth(multipleFSEvaluation,optionString);
                    break;
                default:
                    dataClassifier(multipleFSEvaluation,optionString);
                    break;
            }
            //output features
            PrintWriter tpw = Output.createAppendPrint(optionString+"/time/"+data.relationName());
            for (double time : multipleFSEvaluation.timeMeasures()) {
                tpw.println(time);
            }
            tpw.println(multipleFSEvaluation.timeMeasure());
            tpw.close();
            //output features
            PrintWriter fpw = Output.createAppendPrint(optionString+"/reductions/"+data.relationName());
            for (String reduction : reductions) {
                fpw.println(reduction);
            }
            fpw.close();
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws Exception {
        SemiRepeatableExperiment e = new SemiRepeatableExperiment(5,1,0.2);
        e.setDataFilePath("./dataset");
        TEST test = new TEST();
        test.setOptions(Utils.splitOptions("-D 0.2"));
        Output.setFolder("./output_s/"+test.getClass().getSimpleName()+"/");
        e.setFSAlgorithm(test);
        J48 j48 = new J48();
        IBk iBk = new IBk();
        iBk.setKNN(3);
        Classifier[] cs = new Classifier[]{j48, iBk};
        e.setClassifiers(cs);
        e.run(SemiRepeatableExperiment.DATA_CLASSIFIERS);
    }
}
