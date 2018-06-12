package pers.xin.experiment;

import pers.xin.algorithm.TEST;
import pers.xin.core.attributeSelection.AttributeSelection;
import pers.xin.core.attributeSelection.FSAlgorithm;
import pers.xin.core.evaluation.MultipleFSEvaluation;
import pers.xin.utils.Output;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * run with multiple classifier without repeat feature selection
 * Created by xin on 11/06/2018.
 */
public class RepeatableExperiment {


    public static final int DATA_CLASSIFIER = 1;
    public static final int DATA_CLASSIFIERS = 2;
    public static final int DATA_BOTH = 3;
    public static final int COUNT_CLASSIFIER = 4;
    public static final int COUNT_CLASSIFIERS = 5;
    public static final int COUNT_BOTH = 6;
//    public static final int PARAM_CLASSIFIER = 6;
//    public static final int PARAM_CLASSIFIERS = 7;

    private int numFold = 10;

    private long seed = 1;

    private String dataFilePath;

    private FSAlgorithm m_fsAlgorithm;

    private boolean parallel = true;

    private Classifier[] classifiers;

    private String optionString;

    private String classifiersHeader = "";

    public RepeatableExperiment(int numFold, long seed) {
        this.numFold = numFold;
        this.seed = seed;
    }

    public void setDataFilePath(String dataFilePath) {
        this.dataFilePath = dataFilePath;
    }

    public void setFSAlgorithm(FSAlgorithm fsAlgorithm) {
        this.m_fsAlgorithm = fsAlgorithm;
    }

    public void setClassifiers(Classifier[] classifiers) {
        this.classifiers = classifiers;
    }

    public void run(int fileMode) throws Exception {
        optionString = Utils.joinOptions(m_fsAlgorithm.getOptions());
        for (Classifier classifier : classifiers) {
            classifiersHeader = classifiersHeader + "," + classifier.getClass().getSimpleName();
        }
        String fsAlgorithmName = m_fsAlgorithm.getClass().getSimpleName();
        Output.setFolder("./output/"+fsAlgorithmName+"/");
        if(fileMode==DATA_CLASSIFIERS||fileMode==DATA_BOTH){
            PrintWriter pw = Output.createAppendPrint(optionString);
            pw.println("Data Set"+classifiersHeader);
            pw.close();
        }

        File folder = new File(dataFilePath);
        File[] files = folder.listFiles(f -> f.getName().contains(".arff"));
        List<File> fileArray = Arrays.stream(files).collect(Collectors.toList());
        if(parallel){
            fileArray.parallelStream().forEach(file -> oneDataMultipleClassifierExperiment(file,fileMode));
        }else {

            fileArray.stream().forEach(file -> oneDataMultipleClassifierExperiment(file,fileMode));
        }
    }

    /**
     * output result as table (data,classifier)-pct
     * @param file file
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
            MultipleFSEvaluation multipleFSEvaluation = new MultipleFSEvaluation(data);
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
            PrintWriter fpw = Output.createAppendPrint(optionString+"/reductions/"+data.relationName());
            for (String reduction : reductions) {
                fpw.println(reduction);
            }
            fpw.close();
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    public String getMeasure(Evaluation evaluation){
        return ""+evaluation.pctCorrect();
    }

    private void dataClassifiers(MultipleFSEvaluation multipleFSEvaluation,String paramString) throws Exception {
        //output (data,classifier)->pct
        PrintWriter pw = Output.createAppendPrint(paramString);
        StringBuffer pcts = new StringBuffer();
        for (Classifier classifier : classifiers) {
            Classifier c = AbstractClassifier.makeCopy(classifier);
            Evaluation evaluation = multipleFSEvaluation.nextClassifier(c);
            pcts.append(",");
            pcts.append(getMeasure(evaluation));
        }
        pw.println(multipleFSEvaluation.getHeader().relationName()+pcts.toString());
        pw.close();
    }

    private void dataClassifier(MultipleFSEvaluation multipleFSEvaluation,String paramString) throws Exception {
        for (Classifier classifier : classifiers) {
            Classifier c = AbstractClassifier.makeCopy(classifier);
            Evaluation evaluation = multipleFSEvaluation.nextClassifier(c);
            PrintWriter pw = Output.createAppendPrint(paramString+"/"+classifier.getClass().getSimpleName());
            pw.println(multipleFSEvaluation.getHeader().relationName()+","+getMeasure(evaluation));
            pw.close();
        }
    }

    private void dataBoth(MultipleFSEvaluation multipleFSEvaluation,String paramString) throws Exception {
        //output (data,classifier)->pct
        PrintWriter pw = Output.createAppendPrint(paramString);
        StringBuffer pcts = new StringBuffer();
        for (Classifier classifier : classifiers) {
            Classifier c = AbstractClassifier.makeCopy(classifier);
            Evaluation evaluation = multipleFSEvaluation.nextClassifier(c);
            pcts.append(",");
            pcts.append(getMeasure(evaluation));

            PrintWriter spw = Output.createAppendPrint(paramString+"/"+classifier.getClass().getSimpleName());
            spw.println(multipleFSEvaluation.getHeader().relationName()+","+getMeasure(evaluation));
            spw.close();
        }
        pw.println(multipleFSEvaluation.getHeader().relationName()+pcts.toString());
        pw.close();
    }


    private void countClassifier(MultipleFSEvaluation multipleFSEvaluation,String paramString) throws Exception {
        int maxLength = multipleFSEvaluation.getMinFoldASLength();
        for (int i = 1; i < maxLength; i++) {
            for (Classifier classifier : classifiers) {
                Classifier c = AbstractClassifier.makeCopy(classifier);
                Evaluation evaluation = multipleFSEvaluation.nextClassifier(c,i);
                PrintWriter pw = Output.createAppendPrint(paramString+"/"+
                        multipleFSEvaluation.getHeader().relationName()+"/"+classifier.getClass().getSimpleName());
                pw.println(i+","+getMeasure(evaluation));
                pw.close();
            }
        }
    }

    private void countClassifiers(MultipleFSEvaluation multipleFSEvaluation,String paramString) throws Exception {
        int maxLength = multipleFSEvaluation.getMinFoldASLength();
        PrintWriter pw = Output.createAppendPrint(paramString+"/"+multipleFSEvaluation.getHeader().relationName());
        pw.println("Data Set"+classifiersHeader);
        for (int i = 1; i < maxLength; i++) {
            StringBuffer pcts = new StringBuffer();
            for (Classifier classifier : classifiers) {
                Classifier c = AbstractClassifier.makeCopy(classifier);
                Evaluation evaluation = multipleFSEvaluation.nextClassifier(c,i);
                pcts.append(",");
                pcts.append(getMeasure(evaluation));
            }
            pw.println(i+pcts.toString());
        }
        pw.close();
    }

    private void countBoth(MultipleFSEvaluation multipleFSEvaluation,String paramString) throws Exception {
        int maxLength = multipleFSEvaluation.getMinFoldASLength();
        PrintWriter pw = Output.createAppendPrint(paramString+"/"+multipleFSEvaluation.getHeader().relationName());
        pw.println("Data Set"+classifiersHeader);
        for (int i = 1; i < maxLength; i++) {
            StringBuffer pcts = new StringBuffer();
            for (Classifier classifier : classifiers) {
                Classifier c = AbstractClassifier.makeCopy(classifier);
                Evaluation evaluation = multipleFSEvaluation.nextClassifier(c,i);
                pcts.append(",");
                pcts.append(getMeasure(evaluation));

                PrintWriter spw = Output.createAppendPrint(paramString+"/"+
                        multipleFSEvaluation.getHeader().relationName()+"/"+classifier.getClass().getSimpleName());
                spw.println(i+","+getMeasure(evaluation));
                spw.close();
            }
            pw.println(i+pcts.toString());
        }
        pw.close();
    }

    public static void main(String[] args) throws Exception {
        RepeatableExperiment e = new RepeatableExperiment(10,1);
        e.setDataFilePath("./dataset");
        TEST mrmi = new TEST();
        mrmi.setOptions(Utils.splitOptions("-D 0.2"));
        e.setFSAlgorithm(mrmi);
        J48 j48 = new J48();
        IBk iBk = new IBk();
        iBk.setKNN(3);
        Classifier[] cs = new Classifier[]{j48, iBk};
        e.setClassifiers(cs);
        e.run(RepeatableExperiment.COUNT_BOTH);
    }
}
