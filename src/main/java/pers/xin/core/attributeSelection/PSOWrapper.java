package pers.xin.core.attributeSelection;

import pers.xin.algorithm.RSFSAIDXN;
import pers.xin.core.evaluation.FSEvaluation;
import pers.xin.core.optimization.PSO;
import pers.xin.core.optimization.Position;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

import java.io.File;
import java.io.FileReader;
import java.io.Serializable;
import java.util.List;
import java.util.Random;

/**
 * Created by xin on 03/08/2018.
 */
public class PSOWrapper extends FSAlgorithm implements Serializable{
    private static final long serialVersionUID = 1976134678795860595L;

    protected FSAlgorithm fsAlgorithm;

    protected Classifier classifier;

    protected int maxIterate;

    protected int swarmSize;

    protected double trainRatio;

    protected Random random;

    protected Position bestP;

    protected AbstractFSFunction function;

    public PSOWrapper(int maxIterate, int swarmSize, double trainRatio, Random random) {
        this.maxIterate = maxIterate;
        this.swarmSize = swarmSize;
        this.trainRatio = trainRatio;
        this.random = random;
    }

    public void setFsAlgorithm(FSAlgorithm fsAlgorithm){
        this.fsAlgorithm = fsAlgorithm;
    }

    public void setClassifier(Classifier classifier) {
        this.classifier = classifier;
    }

    public void setFunction(AbstractFSFunction function) {
        this.function = function;
    }

    @Override
    protected List<Integer> select(Instances data) throws Exception {
        if (bestP == null){
//            PctFSFunction function = new PctFSFunction(data,trainRatio,random);
            function.setData(data);
            function.setTrainRatio(trainRatio);
            function.setRandom(random);
            function.setClassifier(classifier);
            function.setFsAlgorithm(fsAlgorithm);
            PSO pso = new PSO(swarmSize,maxIterate,0.6,2,2,random,function);
            bestP = pso.search();
//            System.out.println(bestP);
        }
        ((OptimizableFS)fsAlgorithm).setParams(bestP.get());
        return fsAlgorithm.select(data);
    }

    @Override
    public String getSetting() {
        return null;
    }

    @Override
    public void setOptions(String[] options) throws Exception {

    }

    public static void main(String[] args) throws Exception {
        File file = new File("./dataset/ionosphere.arff");
        Instances data = new Instances(new FileReader((file)));
        data.setClassIndex(data.numAttributes()-1);
        PSOWrapper test = new PSOWrapper(10,20,0.7,new Random());
        test.setClassifier(new J48());
        test.setFsAlgorithm(new RSFSAIDXN());
        test.setFunction(new AucFSFunction());
        AttributeSelection as = new AttributeSelection(test);
        MetaClassifier mc = new MetaClassifier();
        mc.setAttributeSelection(as);
        mc.setClassifier(new J48());
        FSEvaluation evaluation = new FSEvaluation(data);
        Random r = new Random();
        long x = r.nextLong();
        long t1 = System.currentTimeMillis();
        evaluation.crossValidateModel(mc,data,5,new Random(1));
        System.out.println(evaluation.pctCorrect());

        System.out.println(System.currentTimeMillis()-t1);

        Evaluation evaluation1 = new Evaluation(data);
        evaluation1.crossValidateModel(new J48(),data,5,new Random(1));
        System.out.println(evaluation1.pctCorrect());
    }
}
