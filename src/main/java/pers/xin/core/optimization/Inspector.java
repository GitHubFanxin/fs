package pers.xin.core.optimization;

import java.io.Serializable;
import java.math.BigDecimal;
import java.util.Random;

/**
 * inspector in ISO
 * Created by xin on 23/03/2018.
 */
public class Inspector implements Serializable {

    /**
     * Ramdom which decide movement of this inspector.
     */
    private Random random;

    protected int dimension = 1;

    /**
     * keep how many decimal
     */
    private int[] m_precision = new int[]{2};

    /**
     * 惯性权重
     */
    private double w = 0.7;

    /**
     * 学习因子
     */
    private double c1 = 2, c2 = 2;

    /**
     * current currentPosition
     */
    protected Position currentPosition;

    /**
     * current velocity
     */
    protected double[] velocity;

    /**
     * best currentPosition for this inspector
     */
    protected Position pBestPosition;

    /**
     * all inspectors must hold identical positions
     */
    private Positions positions;

    private Function function;


    /**
     * @param w
     * @param c1
     * @param c2
     * @param random
     * @param dimension
     * @param precision
     */
    public Inspector(double w, double c1, double c2, Random random, int dimension, int[] precision) {
        this.random = random;
        this.dimension = dimension;
        this.w = w;
        this.c1 = c1;
        this.c2 = c2;
        m_precision = precision;
        this.random = random;
    }

    /**
     * @param w
     * @param c1
     * @param c2
     * @param random
     * @param dimension
     */
    public Inspector(double w, double c1, double c2, Random random, int dimension) {
        this.random = random;
        this.dimension = dimension;
        this.w = w;
        this.c1 = c1;
        this.c2 = c2;
        m_precision = new int[dimension];
        for (int i = 0; i < dimension; i++) {
            m_precision[i] = 2;
        }
        this.random = random;
    }

    public Inspector(Random random) {
        this.random = random;
    }

    private double formatPosition(double x, int d) {
        double y = x > 0 ? (x < 1 ? x : 1) : 0;
        BigDecimal bg = new BigDecimal(y);
        double returnValue = bg.setScale(m_precision[d], BigDecimal.ROUND_HALF_UP).doubleValue();
        return returnValue;
    }

    private double formatVelocity(double x, int d) {
        return x > -1 ? (x < 1 ? x : 1) : -1;
    }

    protected void move() {
        double[] best = positions.getBestPosition().get();
        double[] pbest = pBestPosition.get();
        double[] current = currentPosition.get();
        for (int i = 0; i < dimension; i++) {
            //update velocity
            velocity[i] = formatVelocity(w * velocity[i] + c1 * random.nextDouble() * (pbest[i] - current[i]) +
                    c2 * random.nextDouble() * (best[i] - current[i]), i);
            //update currentPosition
            current[i] = formatPosition(current[i] + velocity[i], i);
        }
        currentPosition = new Position(current);
        double fitness = function.computeFitness(currentPosition);
        positions.mark(currentPosition, fitness);
        //double check
//        fitness = function.computeFitness(currentPosition);
//        positions.mark(currentPosition,fitness);
        checkPBest();
    }

    public void init(Positions positions, Function function) {
        this.positions = positions;
        this.function = function;
        double[] p = new double[dimension];
        velocity = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            p[i] = formatPosition(random.nextDouble(), i);
            velocity[i] = -1 + 2 * random.nextDouble();
        }
        this.currentPosition = new Position(p);
        double fitness = function.computeFitness(currentPosition);
        positions.mark(currentPosition, fitness);
        this.pBestPosition = new Position(currentPosition);
    }

    public void setPositions(Positions positions) {
        this.positions = positions;
    }

    public void setFunction(Function function) {
        this.function = function;
    }

    public void checkPBest() {
        if (positions.get(currentPosition) > positions.get(pBestPosition)) {
            pBestPosition = new Position(currentPosition);
        }
    }

    public Position getCurrentPosition() {
        return new Position(currentPosition);
    }

    @Override
    public String toString() {
        return "Inspector{" +
                "currentPosition=" + currentPosition +
                ", pBestPosition=" + pBestPosition +
                '}';
    }
}
