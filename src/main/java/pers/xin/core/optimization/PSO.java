package pers.xin.core.optimization;

import java.util.Random;

/**
 * PSO that find a position that has the biggest fitness.
 * Created by xin on 31/03/2018.
 */
public class PSO {

    private int swarmSize;

    private int maxIteration;

    private Positions positions;

    private Particle[] particles;

    private Function function;

    private Random random;

    /**
     * @param swarmSize
     * @param maxIteration
     * @param w
     * @param c1
     * @param c2
     * @param random random generate seed in particles and random for check gBest.
     * @param function
     */
    public PSO(int swarmSize, int maxIteration, double w, double c1, double c2, Random random, Function function) {
        this.swarmSize = swarmSize;
        this.maxIteration = maxIteration;
        this.function = function;
        this.random = random;
        particles = new Particle[swarmSize];
        positions = new Positions();
        for (int i = 0; i < swarmSize; i++) {
            particles[i] = new Particle(w,c1,c2,new Random(this.random.nextLong()), function.dimension());
            particles[i].init(positions,function);
        }
    }

    public Position search(){
        for (int i = 0; i < maxIteration; i++) {
//            System.out.print(".");
            for (int j = 0; j < swarmSize; j++) {
                particles[j].move();
            }
            Position best = positions.getBestPosition();
            double fitness = function.computeFitness(best);
            positions.mark(best,fitness);
//            System.out.println(fitness+"\t"+best+positions.getBestValue());
//            System.out.println("-------");
        }
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
                double value = Math.abs(Math.pow(p[0]-0.5,3))+Math.pow(p[1]-0.25,2)+Math.pow(p[2]-0.3,2)+Math.pow
                        (p[3]-0.2,2);
                return -value;
            }
        };
        PSO pso = new PSO(20,1,0.5,2,2,new Random(),f);
        Position best = pso.search();
        System.out.println(best.toString() + f.computeFitness(best));
    }
}
