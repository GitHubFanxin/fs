package pers.xin.experiment;

import pers.xin.algorithm.TEST;
import pers.xin.core.attributeSelection.AttributeSelection;
import pers.xin.core.attributeSelection.FSAlgorithm;
import pers.xin.core.attributeSelection.MetaClassifier;
import pers.xin.core.evaluation.FSEvaluation;
import pers.xin.core.evaluation.MultipleFSEvaluation;
import pers.xin.utils.ExperimentInfo;
import pers.xin.utils.Output;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.Utils;

import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * Created by xin on 10/06/2018.
 */
public class Experment {

    private boolean parallel = true;

    private boolean repeatable = false;

    private int numFold = 10;

    private long seed = 1;

    private String dataFilePath;

    private FSAlgorithm m_fsAlgorithm;

    private int fileMode = ExperimentInfo.DATA_MEASURE;

    private String[] classifiers = new String[] {
            weka.classifiers.trees.J48.class.getName()
            ,weka.classifiers.lazy.IBk.class.getName()};

    public void setParallel(boolean parallel) {
        this.parallel = parallel;
    }

    public void setRepeatable(boolean repeatable) {
        this.repeatable = repeatable;
    }

    public void setNumFold(int numFold) {
        this.numFold = numFold;
    }

    public void setSeed(long seed) {
        this.seed = seed;
    }

    public void setDataFilePath(String dataFilePath) {
        this.dataFilePath = dataFilePath;
    }

    public void setFsAlgorithm(FSAlgorithm m_fsAlgorithm) {
        this.m_fsAlgorithm = m_fsAlgorithm;
    }

    public void setFileMode(int fileMode) {
        this.fileMode = fileMode;
    }

    public void setClassifiers(String[] classifiers) {
        this.classifiers = classifiers;
    }

    public void run() throws Exception {
        File folder = new File(dataFilePath);
        File[] files = folder.listFiles(f->f.getName().contains(".arff"));
        List<File> fileArray = Arrays.stream(files).collect(Collectors.toList());

        if(parallel){
            if(repeatable){
                if (fileMode==ExperimentInfo.DATA_CLASSIFIER_MEASURE)
                    fileArray.parallelStream().forEach(file -> oneDataMultipleClassifierExperiment(file));
                else{
                    fileArray.parallelStream().forEach(file -> oneDataRepeatableExperiment(file));
                }
            }else {
                if (fileMode==ExperimentInfo.DATA_CLASSIFIER_MEASURE) fileMode = ExperimentInfo.DATA_MEASURE;
                fileArray.parallelStream().forEach(file -> oneDataExperiment(file));
            }
        }else {
            if(repeatable){
                if (fileMode==ExperimentInfo.DATA_CLASSIFIER_MEASURE)
                    fileArray.stream().forEach(file -> oneDataMultipleClassifierExperiment(file));
                else{
                    fileArray.stream().forEach(file -> oneDataRepeatableExperiment(file));
                }
            }else {
                if (fileMode==ExperimentInfo.DATA_CLASSIFIER_MEASURE) fileMode = ExperimentInfo.DATA_MEASURE;
                fileArray.stream().forEach(file -> oneDataExperiment(file));
            }
        }
    }

    public void oneDataExperiment(File file){
        try {
            Instances data = new Instances(new FileReader((file)));
            data.setClassIndex(data.numAttributes()-1);
            String fsAlgorithmName = m_fsAlgorithm.getClass().getSimpleName();
            String paramString = Utils.joinOptions(m_fsAlgorithm.getOptions());

            for (String classifier : classifiers) {
                Classifier c = (Classifier) Class.forName(classifier).newInstance();
                FSAlgorithm copiedAlgorithm = FSAlgorithm.makeCopy(m_fsAlgorithm);
                FSEvaluation evaluation = oneClassifier(c, copiedAlgorithm, data);

                String classifierName = c.getClass().getSimpleName();
                ExperimentInfo ei = new ExperimentInfo(data.relationName(),classifierName,fsAlgorithmName,paramString);
                PrintWriter pw = ei.createAppendPrint(fileMode);
                pw.println(ei.generateLine(evaluation.pctCorrect()+"",fileMode));
                pw.close();

                PrintWriter fpw = ei.createFeaturePrint();
                String[] reductions = evaluation.getCrossValidateReductions();
                for (String reduction : reductions) {
                    fpw.println(reduction);
                }
                fpw.close();
            }
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    /**
     * output result as table (data,classifier)-pct
     * @param file data file
     */
    public void oneDataMultipleClassifierExperiment(File file){
        try {
            Instances data = new Instances(new FileReader((file)));
            data.setClassIndex(data.numAttributes()-1);
            String fsAlgorithmName = m_fsAlgorithm.getClass().getSimpleName();
            String paramString = Utils.joinOptions(m_fsAlgorithm.getOptions());

            FSAlgorithm copiedAlgorithm = FSAlgorithm.makeCopy(m_fsAlgorithm);
            AttributeSelection as = new AttributeSelection(copiedAlgorithm);
            MultipleFSEvaluation multipleFSEvaluation = new MultipleFSEvaluation(data);
            multipleFSEvaluation.initCrossValidate(as,data, numFold,seed);
            String[] reductions = multipleFSEvaluation.getCrossValidateReductions();

            StringBuffer pcts = new StringBuffer();
            for (String classifier : classifiers) {
                Classifier c = (Classifier) Class.forName(classifier).newInstance();
                Evaluation evaluation = multipleFSEvaluation.nextClassifier(c);
                pcts.append(",");
                pcts.append(evaluation.pctCorrect());
            }
            PrintWriter pw = Output.createAppendPrint(paramString);
            pw.println(data.relationName()+pcts.toString());
            pw.close();
            PrintWriter fpw = Output.createAppendPrint(paramString+"/"+data.relationName());
            for (String reduction : reductions) {
                fpw.println(reduction);
            }
            fpw.close();
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    /**
     * output result as table (data,classifier)-pct
     * @param file data file
     */
    public void oneDataRepeatableExperiment(File file){
        try {
            Instances data = new Instances(new FileReader((file)));
            data.setClassIndex(data.numAttributes()-1);
            String fsAlgorithmName = m_fsAlgorithm.getClass().getSimpleName();
            String paramString = Utils.joinOptions(m_fsAlgorithm.getOptions());

            FSAlgorithm copiedAlgorithm = FSAlgorithm.makeCopy(m_fsAlgorithm);
            AttributeSelection as = new AttributeSelection(copiedAlgorithm);
            MultipleFSEvaluation multipleFSEvaluation = new MultipleFSEvaluation(data);
            multipleFSEvaluation.initCrossValidate(as,data,numFold,seed);
            String[] reductions = multipleFSEvaluation.getCrossValidateReductions();

            StringBuffer pcts = new StringBuffer();
            for (String classifier : classifiers) {
                Classifier c = (Classifier) Class.forName(classifier).newInstance();
                Evaluation evaluation = multipleFSEvaluation.nextClassifier(c);

                String classifierName = c.getClass().getSimpleName();
                ExperimentInfo ei = new ExperimentInfo(data.relationName(),classifierName,fsAlgorithmName,paramString);
                PrintWriter pw = ei.createAppendPrint(fileMode);
                pw.println(ei.generateLine(evaluation.pctCorrect()+"",fileMode));
                pw.close();
                PrintWriter fpw = ei.createFeaturePrint();
                for (String reduction : reductions) {
                    fpw.println(reduction);
                }
                fpw.close();
            }
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    public FSEvaluation oneClassifier(Classifier baseClassifier, FSAlgorithm algorithm, Instances data) throws Exception {
        MetaClassifier c = new MetaClassifier();
        AttributeSelection as = new AttributeSelection(algorithm);
        c.setAttributeSelection(as);
        c.setClassifier(baseClassifier);
        FSEvaluation evaluation = new FSEvaluation(data);
        evaluation.crossValidateModel(c,data,numFold,new Random(seed));
        return evaluation;
    }

    public static void main(String[] args) throws Exception {
        Experment e = new Experment();
        e.setDataFilePath("./dataset");
        e.setRepeatable(true);
        e.setFileMode(ExperimentInfo.DATA_MEASURE);
        TEST mrmi = new TEST();
        mrmi.setOptions(Utils.splitOptions("-D 0.2"));
        e.setFsAlgorithm(mrmi);
        e.run();
    }


}
