package pers.xin.utils;

import pers.xin.core.evaluation.FSEvaluation;

import java.io.PrintWriter;

/**
 * generate path based on organization of record file.
 *
 * <pre>
 * DATA_MEASURE: data - measure per row
 * COUNT_MEASURE: count - measure per row
 * PARAM_MEASURE: param - measure per row
 * FEATURE: reduction per row
 * </pre>
 * Created by xin on 10/06/2018.
 */
public class ExperimentInfo {
    public static final int DATA_MEASURE = 1;
    public static final int COUNT_MEASURE = 2;
    public static final int PARAM_MEASURE = 3;
    public static final int DATA_CLASSIFIER_MEASURE = 4;

    private String data;
    private String classifier;
    private String fsAlgorithm;
    private String params;

    public ExperimentInfo(String data, String classifier, String fsAlgorithm, String params) {
        this.data = data;
        this.classifier = classifier;
        this.fsAlgorithm = fsAlgorithm;
        this.params = params;
    }

    private String format(int fileMode){
        switch (fileMode){
            case DATA_MEASURE:
                return params+"/"+classifier;
            case COUNT_MEASURE:
                return params+"/"+data+"/"+classifier;
            case PARAM_MEASURE:
                return data+"/"+classifier;
            case DATA_CLASSIFIER_MEASURE:
                return params;
            default:
                return null;
        }
    }

    public String featureFormat(){
        return params+"/"+data;
    }

    public String generateLine(String measure, int fileMode, String... other){
        switch (fileMode){
            case DATA_MEASURE:
                return data+","+measure;
            case COUNT_MEASURE:
                return other[0]+","+measure;
            case PARAM_MEASURE:
                return params+","+measure;
            case DATA_CLASSIFIER_MEASURE:
                return data+measure;
            default:
                return null;
        }
    }

    public PrintWriter createPrint(int fileMode) throws Exception {
        return Output.createPrint(format(fileMode));
    }

    public PrintWriter createAppendPrint(int fileMode) throws Exception {
        return Output.createAppendPrint(format(fileMode));
    }

    public PrintWriter createFeaturePrint() throws Exception {
        return Output.createPrint(featureFormat());
    }
}
