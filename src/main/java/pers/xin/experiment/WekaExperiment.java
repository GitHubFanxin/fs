package pers.xin.experiment;

import pers.xin.core.attributeSelection.WekaMetaClassifier;
import pers.xin.core.evaluation.FSEvaluation;
import pers.xin.utils.Output;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SerializedObject;

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
public class WekaExperiment {
    public static final int DATA_CLASSIFIER = 1;

    private int numFold = 10;

    private long seed = 1;

    private String dataFilePath;

    private boolean parallel = true;

    private ASEvaluation asEvaluation;

    private ASSearch asSearch;

    private Classifier[] classifiers;

    private String classifiersHeader = "";

    String fsAlgorithmName = "weka";

    public WekaExperiment(int numFold, long seed) {
        this.numFold = numFold;
        this.seed = seed;
    }

    public void setFsAlgorithmName(String fsAlgorithmName) {
        this.fsAlgorithmName = fsAlgorithmName;
    }

    public void setAS(ASEvaluation asEvaluation, ASSearch asSearch){
        this.asEvaluation = asEvaluation;
        this.asSearch = asSearch;
        fsAlgorithmName = asEvaluation.getClass().getSimpleName()+"+"+asSearch.getClass().getSimpleName();
    }


    public void setParallel(boolean parallel) {
        this.parallel = parallel;
    }

    public void setDataFilePath(String dataFilePath) {
        this.dataFilePath = dataFilePath;
    }

    public void setClassifiers(Classifier[] classifiers) {
        this.classifiers = classifiers;
    }

    public void run() throws Exception {
        for (Classifier classifier : classifiers) {
            classifiersHeader = classifiersHeader + "," + classifier.getClass().getSimpleName();
        }
        Output.setFolder("./output/"+fsAlgorithmName+"/");

        File folder = new File(dataFilePath);
        File[] files = folder.listFiles(f -> f.getName().contains(".arff"));
        List<File> fileArray = Arrays.stream(files).collect(Collectors.toList());
        if(parallel){
            fileArray.parallelStream().forEach(file -> oneDataMultipleClassifierExperiment(file));
        }else {
            fileArray.stream().forEach(file -> oneDataMultipleClassifierExperiment(file));
        }

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
                FSEvaluation evaluation = new FSEvaluation(data);
                WekaMetaClassifier classifierWithFS = new WekaMetaClassifier();
                ASEvaluation copiedASEvaluation = (ASEvaluation) new SerializedObject(asEvaluation).getObject();
                ASSearch copideASSearch = (ASSearch) new SerializedObject(asSearch).getObject();
                classifierWithFS.setEvaluator(copiedASEvaluation);
                classifierWithFS.setSearch(copideASSearch);
                evaluation.crossValidateModel(classifierWithFS,data,numFold,new Random(seed));

                PrintWriter pw = Output.createAppendPrint(classifier.getClass().getSimpleName());
                pw.println(evaluation.getHeader().relationName()+","+getMeasure(evaluation));
                pw.close();

                String[] reductions = evaluation.getCrossValidateReductions();
                //output features
                PrintWriter fpw = Output.createAppendPrint("reductions/"+
                        classifier.getClass().getSimpleName()+"/"+data.relationName());
                for (String reduction : reductions) {
                    fpw.println(reduction);
                }
                fpw.close();

                //output features
                PrintWriter tpw = Output.createAppendPrint("time/"+
                        classifier.getClass().getSimpleName()+"/"+data.relationName());
                for (double time : evaluation.timeMeasures()) {
                    tpw.println(time);
                }
                tpw.println(evaluation.timeMeasure());
                tpw.close();
            }

        } catch (Exception e){
            e.printStackTrace();
        }
    }

    public String getMeasure(Evaluation evaluation){
        return ""+evaluation.pctCorrect();
    }

    public static void main(String[] args) throws Exception {
        WekaExperiment e = new WekaExperiment(10,1);
        e.setDataFilePath("./dataset");
        e.setAS(new CfsSubsetEval(),new GreedyStepwise());
        J48 j48 = new J48();
        IBk iBk = new IBk();
        iBk.setKNN(3);
        Classifier[] cs = new Classifier[]{j48, iBk};
        e.setClassifiers(cs);
        e.setParallel(false);
        e.run();
    }
}
