package pers.xin.core.optimization;

import java.util.Random;

/**
 * PSO that find a position that has the biggest fitness.
 * Created by xin on 31/03/2018.
 */
public class ISO {

    private int swarmSize;

    private int maxIteration;

    private Positions positions;

    private Inspector[] inspectors;

    private Function function;

    private Random random;

    /**
     * @param swarmSize
     * @param maxIteration
     * @param w
     * @param c1
     * @param c2
     * @param random random generate seed in inspectors and random for check gBest.
     * @param function
     */
    public ISO(int swarmSize, int maxIteration, double w, double c1, double c2,Random random, Function function) {
        this.swarmSize = swarmSize;
        this.maxIteration = maxIteration;
        this.function = function;
        this.random = random;
        inspectors = new Inspector[swarmSize];
        positions = new Positions();
        for (int i = 0; i < swarmSize; i++) {
            inspectors[i] = new Inspector(w,c1,c2,new Random(this.random.nextLong()), function.dimension());
            inspectors[i].init(positions,function);
        }
    }

    public Position search(){
        for (int i = 0; i < maxIteration; i++) {
            for (int j = 0; j < swarmSize; j++) {
                inspectors[j].move();
            }
            Position best = positions.getBestPosition();
            double fitness = function.computeFitness(best);
            positions.mark(best,fitness);
            System.out.println(fitness+"\t"+best+positions.getBestValue());
            System.out.println("-------");
        }


//        for (int i = 0; i < maxIteration; i++) {
//            Position best = positions.getBestPosition();
//            double fitness = function.computeFitness(best);
//            positions.mark(best,fitness);
//        }

        return positions.getBestPosition();
    }

    public static void main(String[] args) {
        Function f = new Function() {
            @Override
            public int dimension() {
                return 4;
            }

            @Override
            public double computeFitness(Position params) {
                double[] p = params.get();
                double value = Math.pow(p[0]-0.11,2)+Math.pow(p[1]-0.25,2)+Math.pow(p[2]-0.3,2)+Math.pow(p[3]-0.2,2);
                return -value;
            }
        };
        ISO iso = new ISO(20,20,0.5,2,2,new Random(),f);
        Position best = iso.search();
        System.out.println(best.toString() + f.computeFitness(best));
    }
}
